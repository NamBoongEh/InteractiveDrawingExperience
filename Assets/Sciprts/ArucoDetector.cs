using UnityEngine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenCvSharp.Aruco;
using ArucoDict = OpenCvSharp.Aruco.Dictionary;
#if UNITY_SENTIS
using Unity.Sentis;
#endif

public class ArucoDetector : MonoBehaviour
{
    [Header("Settings")]
    public float holdDurationToCapture = 1.5f;

    // 출력 크기 고정 상수: 자연 크기 582×353 기준 ×4 배율 (비율 1.649:1 유지)
    private const int kOutputW = 2328;
    private const int kOutputH = 1412;

    [Header("Physical Dimensions (mm) — 비율 기반 마스크 크기")]
#if UNITY_SENTIS
    [Header("AI Segmentation (Unity Sentis)")]
    public ModelAsset aiSegModel;
    [Tooltip("추론 백엔드. GPUCompute = GPU(빠름), CPU = 호환성 높음.")]
    public BackendType aiBackend = BackendType.GPUCompute;
    int aiInputSize = 320;
    [Tooltip("살리언시 이진화 임계값. 높을수록 마스크 좁아짐 (0.4~0.6 권장).")]
    public float aiThreshold = 0.0341f;

    [Tooltip("중앙 가우시안 σ. 외부 그림 억제 강도. 0=비활성화, 0.35=권장.")]
    public float aiCenterSigma = 0.041f;
    [Tooltip("마스크 침식(px). 마스크가 그림보다 크게 나올 때 줄임 (8~15 권장).")]
    public int aiMaskErodePx = 1;
    [Tooltip("꼬리 내부 구멍(hollow) 자동 채우기. 꼬리 안쪽이 투명해지는 문제 해결.")]
    public bool aiHoleFill = true;
#else
    [Header("AI Segmentation  ※ com.unity.sentis 패키지 미설치")]
    [Tooltip("Package Manager → Add by name → com.unity.sentis 설치 후 사용 가능.")]
#endif

    [Header("Ink Line Clip (2차 가공 — 외곽선 내부 정밀 클리핑)")]
    [Tooltip("세그멘테이션 결과에서 검정 외곽선 내부만 남기는 2차 정밀 클리핑.\n" +
             "외곽선 밖 잔여 픽셀 제거. AI/컨투어/템플릿 모든 경로에 적용.")]
    public bool useInkLineClip = true;
    [Range(50, 150)]
    [Tooltip("외곽선(잉크) 픽셀 V 임계값. V < 이 값 = 검정 잉크로 판정.\n" +
             "alpha 경계를 이 값 이하의 어두운 픽셀까지 확장. 권장 80~100.\n" +
             "※ 30 이하는 검정 잉크 미검출 위험 — 최솟값 50으로 제한.")]
    public int inkInkThresh = 80;

    [Header("Mask Offset (픽셀 단위 마스크 위치 보정)")]
    [Tooltip("fish별 방향별 마스크 오프셋 (픽셀). x=좌(-)/우(+), y=위(-)/아래(+).\n" +
             "아루코 마커 방향(정면/좌90/180/우90)에 따라 각각 다른 오프셋 적용.")]
    public MaskOffsetSet[] maskOffsets = new MaskOffsetSet[]
    {
        new MaskOffsetSet(),   // fish0: 모든 방향 0,0
    };

    [Header("Upscayl Settings")]
    [Tooltip("StreamingAssets 기준 upscayl-bin 실행 파일 상대 경로.\n" +
             "설치 위치: StreamingAssets/upscayl-ncnn/ 폴더에 exe + models/ 폴더를 배치.")]
    public string upscaylExePath = "upscayl-ncnn/upscayl-bin.exe";
    [Tooltip("사용할 모델 이름 (models 폴더 내 파일명, 확장자 제외).\n" +
             "digital-art-4x: 스케치/그림 (권장)\n" +
             "upscayl-standard-4x: 범용\n" +
             "high-fidelity-4x: 고품질 디테일")]
    public string upscaylModelName = "digital-art-4x";

    [Header("Debug Save")]
    [Tooltip("true 시 각 처리 단계 이미지를 Assets/Resources/FishTest/ 에 저장합니다.")]
    public bool saveDebugImages = false;

    [Header("Debug UI")]
    public UnityEngine.UI.RawImage overlayView;
    public UnityEngine.UI.Text failureText;

    public event Action<Texture2D, int> OnAllMarkersDetected;

    private WebCamCapture cam;
    private ArucoDict arucoDict;
    private DetectorParameters detParams;
    private float holdTimer = 0f;

    // ── 리소스 경로 상수 ──────────────────────────────────────
    private const string kApplierPrefix  = "FishMaskApplier/fish";
    private const string kFishPrefix     = "Fish/fish";
    private const int    kMaxFishCount   = 20;

    // ── FishMaskApplier/ 폴더: AI 세그멘테이션 후 AND 마스크 적용
    // fishApplierSources.Length = fish 수 (fishCount 필드 불필요)
    private Texture2D[] fishApplierSources;
    private Mat[]       _fishApplierMasks;   // Start()에서 1회 변환·캐시

    private int         _fishCount;     // fishApplierSources.Length 캐시 (텍스처 해제 후 참조용)

    // ── GC 감소용 정적 재사용 상수 ────────────────────────────
    // MixChannels fromTo 배열: 그레이스케일 ch0 → RGBA ch3(알파) 복사
    private static readonly int[]   s_alphaMixMap = { 0, 3 };
    // ContourFillHoles FloodFill 시드 버퍼 (크기 고정, 값만 매번 갱신)
    private static readonly Point[] s_holeSeeds   = new Point[4];

    // ── Unity Sentis ───────────────────────────────────────────
#if UNITY_SENTIS
    private Worker _sentisWorker;       // U2Net
#endif

    // ── 재사용 버퍼 ───────────────────────────────────────────
    private byte[]    _overlayBuffer;
    private Texture2D _overlayTex;
    private readonly Dictionary<int, int> _idToIdx = new Dictionary<int, int>();
    private byte[]    _matToTexBuf;       // MatToTexture2D 재사용 버퍼 (~11 MB)
#if UNITY_SENTIS
    private byte[]    _aisegRawBuf;       // AISeg 전처리 원시 바이트 버퍼
    private float[]   _aisegFloatBuf;     // AISeg 전처리 float 버퍼
    private byte[]    _aisegPixelBuf;     // AISeg_SaliencyToMat 결과 픽셀 버퍼
#endif

    private static readonly string[] s_rotLabels = { "회전 없음", "90°CCW", "180°", "90°CW" };
    private bool _isProcessing = false;

    // ── 초기화 ────────────────────────────────────────────────
    void Start()
    {
        cam = ScannerManager.Instance.webCam;
        arucoDict = CvAruco.GetPredefinedDictionary(PredefinedDictionaryName.Dict4X4_50);
        detParams = DetectorParameters.Create();
        SetFailureMessage(null);

        fishApplierSources = LoadTextureSequence(kApplierPrefix, kMaxFishCount);
        if (fishApplierSources.Length == 0)
        {
            int count = CountResources(kFishPrefix, kMaxFishCount);
            fishApplierSources = new Texture2D[count]; // 전부 null, 크기만 맞춤
        }
        else
        {
            for (int i = 0; i < fishApplierSources.Length; i++)
                ManagerScripts.Log($"[ArucoDetector] FishMaskApplier fish{i}: {fishApplierSources[i].width}×{fishApplierSources[i].height}");
        }

        _fishCount        = fishApplierSources.Length;
        _fishApplierMasks = new Mat[_fishCount];
        for (int i = 0; i < _fishCount; i++)
            if (fishApplierSources[i] != null)
                _fishApplierMasks[i] = ApplierTextureToMask(fishApplierSources[i]);

        foreach (var tex in fishApplierSources)
            if (tex != null) Resources.UnloadAsset(tex);
        fishApplierSources = null;

        if (_fishCount == 0)
        {
            ManagerScripts.Log("[W] [ArucoDetector] Fish 리소스가 없습니다. FishMaskApplier/ 또는 Fish/ 폴더를 확인하세요.");
            return;
        }
        ManagerScripts.Log($"[ArucoDetector] Fish {_fishCount}개 로드 → ID 범위 0~{_fishCount * 4 - 1}");

        // ── Sentis 초기화 (U2Net) ────────────────────────────────
#if UNITY_SENTIS
            if (aiSegModel == null)
                Debug.LogError("[AISeg] aiSegModel이 Inspector에 할당되지 않았습니다.");
            else
            {
                try
                {
                    _sentisWorker = new Worker(ModelLoader.Load(aiSegModel), aiBackend);
                    Debug.Log($"[AISeg] U2Net Worker 초기화 완료 (backend={aiBackend}  inputSize={aiInputSize})");
                }
                catch (Exception e) { Debug.LogError($"[AISeg] Worker 초기화 실패: {e.Message}"); }
            }
#endif
    }


    void OnDestroy()
    {
        if (_overlayTex != null) Destroy(_overlayTex);
#if UNITY_SENTIS
        _sentisWorker?.Dispose();
#endif
        if (_fishApplierMasks != null)
            foreach (var m in _fishApplierMasks) m?.Dispose();
    }

    // ── 매 프레임 감지 ────────────────────────────────────────
    void Update()
    {
        if (ScannerManager.Instance.State != ScannerState.Scanning) return;
        if (!cam.IsPlaying) return;

        using Mat frame = cam.GrabFrameRGBA();
        if (frame == null) return;
        using Mat gray  = new Mat();
        Cv2.CvtColor(frame, gray, ColorConversionCodes.RGBA2GRAY);

        var (found, corners, ids) = TryDetect(gray);

        if (found)
        {
            DrawOverlay(frame, corners, ids);
            holdTimer += Time.deltaTime;

            if (holdTimer >= holdDurationToCapture && !_isProcessing)
            {
                holdTimer = 0f;
                _isProcessing = true;
                Mat frameCopy = frame.Clone();
                _ = WarpAndFireAsync(frameCopy, corners, ids);
            }
        }
        else
        {
            holdTimer = 0f;
        }
    }

    async Task WarpAndFireAsync(Mat frameCopy, Point2f[][] corners, int[] ids)
    {
        try
        {
            var (warped, tlMarkerId) = await WarpAsync(frameCopy, corners, ids);
            SetFailureMessage(null);
            OnAllMarkersDetected?.Invoke(warped, tlMarkerId);
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError($"[ArucoDetector] 이미지 생성 실패: {e.Message}");
            SetFailureMessage("이미지 생성에 실패했습니다.\n다시 카메라에 인식해 주세요.");
        }
        finally
        {
            frameCopy.Dispose();
            _isProcessing = false;
        }
    }

    // ── upscayl-bin CLI 4× AI 업스케일 ──────────────────────────
    // ① 메인 스레드: Mat → PNG bytes (Texture2D 사용)
    // ② 백그라운드: 파일 저장 + 프로세스 실행 + 결과 읽기
    // ③ 메인 스레드: PNG bytes → Mat (Texture2D 사용)
    async Task<Mat> UpscaylAsync(Mat input)
    {
        string tempDir   = Application.temporaryCachePath;
        string inPath    = Path.Combine(tempDir, "upscayl_in.png");
        string outPath   = Path.Combine(tempDir, "upscayl_out.png");
        string exeFull   = Path.Combine(Application.streamingAssetsPath, upscaylExePath);
        string modelsDir = Path.Combine(Path.GetDirectoryName(exeFull), "models");

        // ① 메인 스레드에서 PNG 인코딩
        byte[] inBytes = MatToPngBytes(input);

        // ② 파일 I/O + 프로세스 실행 (백그라운드)
        string stderr   = "";
        int    exitCode = 0;
        byte[] outBytes = null;
        await Task.Run(() =>
        {
            File.WriteAllBytes(inPath, inBytes);
            var psi = new System.Diagnostics.ProcessStartInfo
            {
                FileName               = exeFull,
                Arguments              = $"-i \"{inPath}\" -o \"{outPath}\" -s 4 -n {upscaylModelName} -m \"{modelsDir}\"",
                UseShellExecute        = false,
                CreateNoWindow         = true,
                RedirectStandardError  = true,
                RedirectStandardOutput = true,
            };
            using var proc = System.Diagnostics.Process.Start(psi);
            proc.WaitForExit();
            stderr   = proc.StandardError.ReadToEnd();
            exitCode = proc.ExitCode;
            if (exitCode == 0)
                outBytes = File.ReadAllBytes(outPath);
        });

        if (exitCode != 0)
            throw new Exception($"upscayl-bin 실패 (code={exitCode}): {stderr}");

        // ③ 메인 스레드에서 PNG 디코딩
        Mat result = PngBytesToMat(outBytes);
        if (result == null || result.Empty())
            throw new Exception("upscayl-bin 출력 이미지 로드 실패");

        UnityEngine.Debug.Log($"[UpscaylAsync] {input.Cols}×{input.Rows} → {result.Cols}×{result.Rows}");
        return result;
    }

    // Mat → PNG byte[] (메인 스레드 전용 — Texture2D 사용)
    byte[] MatToPngBytes(Mat mat)
    {
        int W = mat.Cols, H = mat.Rows;
        Mat src = mat;
        bool needDispose = false;
        if (mat.Channels() != 4)
        {
            src = new Mat();
            Cv2.CvtColor(mat, src, ColorConversionCodes.BGR2RGBA);
            needDispose = true;
        }
        using Mat flipped = new Mat();
        Cv2.Flip(src, flipped, FlipMode.X);
        if (needDispose) src.Dispose();

        byte[] buf = new byte[W * H * 4];
        Marshal.Copy(flipped.Data, buf, 0, buf.Length);
        Texture2D tex = new Texture2D(W, H, TextureFormat.RGBA32, false);
        tex.LoadRawTextureData(buf);
        tex.Apply();
        byte[] png = tex.EncodeToPNG();
        DestroyImmediate(tex);
        return png;
    }

    // PNG byte[] → RGBA Mat (메인 스레드 전용 — Texture2D 사용)
    Mat PngBytesToMat(byte[] pngBytes)
    {
        Texture2D tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
        tex.LoadImage(pngBytes);
        int W = tex.width, H = tex.height;
        Color32[] pixels = tex.GetPixels32();
        DestroyImmediate(tex);

        byte[] buf = new byte[W * H * 4];
        for (int i = 0; i < pixels.Length; i++)
        {
            buf[i * 4 + 0] = pixels[i].r;
            buf[i * 4 + 1] = pixels[i].g;
            buf[i * 4 + 2] = pixels[i].b;
            buf[i * 4 + 3] = pixels[i].a;
        }
        Mat result = new Mat(H, W, MatType.CV_8UC4);
        Marshal.Copy(buf, 0, result.Data, buf.Length);
        Cv2.Flip(result, result, FlipMode.X);
        return result;
    }

    void SetFailureMessage(string message)
    {
        if (failureText == null) return;
        bool has = !string.IsNullOrEmpty(message);
        failureText.text = has ? message : string.Empty;
        failureText.gameObject.SetActive(has);
    }

    // ── RGBA Mat → Texture2D ──────────────────────────────────
    Texture2D MatToTexture2D(Mat mat)
    {
        using Mat rgba = new Mat();
        if (mat.Channels() == 3)
            Cv2.CvtColor(mat, rgba, ColorConversionCodes.BGR2RGBA);
        else
            mat.CopyTo(rgba);
        Cv2.Flip(rgba, rgba, FlipMode.X);
        int byteLen = rgba.Rows * rgba.Cols * 4;
        if (_matToTexBuf == null || _matToTexBuf.Length != byteLen)
            _matToTexBuf = new byte[byteLen];
        Marshal.Copy(rgba.Data, _matToTexBuf, 0, byteLen);
        Texture2D tex = new Texture2D(rgba.Cols, rgba.Rows, TextureFormat.RGBA32, false);
        tex.LoadRawTextureData(_matToTexBuf);
        tex.Apply();
        return tex;
    }

    // ── ArUco 마커 감지 ───────────────────────────────────────
    (bool, Point2f[][], int[]) TryDetect(Mat gray)
    {
        CvAruco.DetectMarkers(gray, arucoDict,
            out Point2f[][] allCorners, out int[] allIds, detParams, out _);

        if (allIds == null || allIds.Length < 4)
            return (false, null, null);

        _idToIdx.Clear();
        for (int i = 0; i < allIds.Length; i++)
            _idToIdx[allIds[i]] = i;

        for (int g = 0; g < _fishCount; g++)
        {
            int b = g * 4;
            if (!_idToIdx.ContainsKey(b) || !_idToIdx.ContainsKey(b+1) ||
                !_idToIdx.ContainsKey(b+2) || !_idToIdx.ContainsKey(b+3))
                continue;
            return (true,
                new Point2f[4][] {
                    allCorners[_idToIdx[b  ]], allCorners[_idToIdx[b+1]],
                    allCorners[_idToIdx[b+2]], allCorners[_idToIdx[b+3]]
                },
                new int[] { b, b+1, b+2, b+3 });
        }
        return (false, null, null);
    }

    // ── 마커 중심 + 전체 중점 계산 ────────────────────────────
    static (Point2f[] centers, float midX, float midY) ComputeCenters(Point2f[][] corners, int count)
    {
        var centers = new Point2f[count];
        float sumX = 0, sumY = 0;
        for (int i = 0; i < count; i++)
        {
            var c = corners[i];
            centers[i] = new Point2f(
                (c[0].X+c[1].X+c[2].X+c[3].X)/4f,
                (c[0].Y+c[1].Y+c[2].Y+c[3].Y)/4f);
            sumX += centers[i].X; sumY += centers[i].Y;
        }
        return (centers, sumX/count, sumY/count);
    }

    static Point2f ClosestCorner(Point2f[] c, float tx, float ty)
    {
        Point2f best = c[0]; float minD = float.MaxValue;
        foreach (var p in c) { float d=(p.X-tx)*(p.X-tx)+(p.Y-ty)*(p.Y-ty); if(d<minD){minD=d;best=p;} }
        return best;
    }

    // ── 내부 꼭짓점 + 회전 오프셋 + TL 마커 ID 동시 계산 (ComputeCenters 1회) ──
    static (Point2f[] innerCorners, int anchorOffset, int tlMarkerId) GetWarpParams(Point2f[][] corners, int[] ids)
    {
        var (centers, midX, midY) = ComputeCenters(corners, ids.Length);
        Point2f tl=default, tr=default, br=default, bl=default;
        int anchorOffset = 0;
        int tlMarkerId = ids[0];

        for (int i = 0; i < ids.Length; i++)
        {
            bool left = centers[i].X <= midX, top = centers[i].Y <= midY;
            Point2f inner = ClosestCorner(corners[i], midX, midY);

            if      ( left &&  top) { tl = inner; tlMarkerId = ids[i]; if (ids[i] % 4 == 0) anchorOffset = 0; }
            else if (!left &&  top) { tr = inner;                      if (ids[i] % 4 == 0) anchorOffset = 1; }
            else if (!left && !top) { br = inner;                      if (ids[i] % 4 == 0) anchorOffset = 2; }
            else                    { bl = inner;                      if (ids[i] % 4 == 0) anchorOffset = 3; }
        }
        return (new[] { tl, tr, br, bl }, anchorOffset, tlMarkerId);
    }

    // ── 워프 → AI 업스케일 → 회전 → 마스크 ──────────────────────
    // 자연 크기(582×353)로 워프 후 upscayl-bin 4× AI 업스케일 → 2328×1412
    async Task<(Texture2D tex, int tlMarkerId)> WarpAsync(Mat frame, Point2f[][] corners, int[] ids)
    {
        var (src, anchorPos, tlMarkerId) = GetWarpParams(corners, ids);
        int fishIndex = tlMarkerId / 4;

        // ① 자연 크기(582×353)로 워프 — AI가 4× 복원하여 kOutputW×kOutputH(2328×1412)로 맞춤
        int warpW = 582, warpH = 353;
        Point2f[] dst = {
            new Point2f(0,     0),
            new Point2f(warpW, 0),
            new Point2f(warpW, warpH),
            new Point2f(0,     warpH)
        };
        using Mat M = Cv2.GetPerspectiveTransform(src, dst);
        Mat rawWarp = new Mat();
        Cv2.WarpPerspective(frame, rawWarp, M, new OpenCvSharp.Size(warpW, warpH),
                            InterpolationFlags.Lanczos4, BorderTypes.Constant,
                            new Scalar(255, 255, 255, 255));

        SaveDebugMat(rawWarp, "00_rawWarp", fishIndex);
        Debug.Log($"[Warp] ① rawWarp: {rawWarp.Cols}×{rawWarp.Rows}  (fish{fishIndex})");

        // ② upscayl-bin 4× AI 업스케일 (백그라운드 스레드, 메인 스레드 비차단)
        Mat upscaled;
        try   { upscaled = await UpscaylAsync(rawWarp); }
        finally { rawWarp.Dispose(); }
        Debug.Log($"[Warp] ② upscaled: {upscaled.Cols}×{upscaled.Rows}");
        SaveDebugMat(upscaled, "00_warp_raw", fishIndex);

        if (anchorPos == 0)
        {
            DispatchSegmentation(upscaled, fishIndex, anchorPos: 0);
            SaveDebugMat(upscaled, "07_after_mask", fishIndex);
            Debug.Log($"[Warp] 최종 출력: {upscaled.Cols}×{upscaled.Rows} ({s_rotLabels[0]})");
            var result0 = (FinalizeOutput(upscaled, fishIndex), tlMarkerId);
            upscaled.Dispose();
            return result0;
        }

        RotateFlags flag = anchorPos == 1 ? RotateFlags.Rotate90CounterClockwise
                         : anchorPos == 2 ? RotateFlags.Rotate180
                         :                  RotateFlags.Rotate90Clockwise;
        Mat rotated = new Mat();
        Cv2.Rotate(upscaled, rotated, flag);
        upscaled.Dispose();

        DispatchSegmentation(rotated, fishIndex, anchorPos);
        SaveDebugMat(rotated, "07_after_mask", fishIndex);
        Debug.Log($"[Warp] 최종 출력: {rotated.Cols}×{rotated.Rows} ({s_rotLabels[anchorPos]})");
        var result = (FinalizeOutput(rotated, fishIndex), tlMarkerId);
        rotated.Dispose();
        return result;
    }

    // ── Texture2D 변환 ─────────────────────────────────────────
    // ApplyMask(Otsu + 컨투어 내부 채우기) alpha 기준 tight crop → 2× 업스케일 → 반환
    Texture2D FinalizeOutput(Mat mat, int fishIndex)
    {
        SaveDebugMat(mat, "08_final", fishIndex);

        // ── alpha 채널 추출 및 진단 ───────────────────────────────
        if (mat.Channels() < 4)
        {
            Debug.LogWarning($"[FinalizeOutput] mat가 {mat.Channels()}ch — alpha 없음, 그대로 반환");
            return MatToTexture2D(mat);
        }

        using Mat a = new Mat();
        Cv2.ExtractChannel(mat, a, 3);
        SaveDebugMat(a, "08a_alpha", fishIndex); // ← 알파 채널 상태 확인용

        // alpha 통계: 모두 0이면 MixChannels 실패, 모두 255면 full-opaque
        double minVal, maxVal;
        Cv2.MinMaxLoc(a, out minVal, out maxVal);
        Debug.Log($"[FinalizeOutput] alpha min={minVal} max={maxVal}  ch={mat.Channels()}  fish{fishIndex}");

        using Mat nz = new Mat();
        Cv2.FindNonZero(a, nz);

        // alpha가 모두 0(MixChannels 실패) 또는 모두 255(미설정)인 경우: inkBBox 기반 fallback crop
        bool alphaBad = (nz == null || nz.Empty()) || maxVal == 0 || minVal == maxVal;

        OpenCvSharp.Rect cropRect;
        if (!alphaBad)
        {
            cropRect = Cv2.BoundingRect(nz);
            Debug.Log($"[FinalizeOutput] alpha bbox=({cropRect.X},{cropRect.Y}) {cropRect.Width}×{cropRect.Height}  fish{fishIndex}");
        }
        else
        {
            // alpha 추출 실패 → 전체 캔버스를 crop rect로 사용 (margin 없음)
            Debug.LogWarning($"[FinalizeOutput] alpha 이상(min={minVal} max={maxVal}) → 전체 캔버스 사용  fish{fishIndex}");
            cropRect = new OpenCvSharp.Rect(0, 0, mat.Cols, mat.Rows);
        }

        int pad = 3; // 최소 패딩만 사용
        int x = Mathf.Max(0, cropRect.X - pad);
        int y = Mathf.Max(0, cropRect.Y - pad);
        int w = Mathf.Min(mat.Cols - x, cropRect.Width  + pad * 2);
        int h = Mathf.Min(mat.Rows - y, cropRect.Height + pad * 2);

        if (w > 0 && h > 0)
        {
            using Mat cropped = new Mat(mat, new OpenCvSharp.Rect(x, y, w, h));

            // ── RGB 언샤프마스킹 (AI 업스케일 후 선명도 보정) ──────
            using Mat croppedBgr = new Mat();
            Cv2.CvtColor(cropped, croppedBgr, ColorConversionCodes.RGBA2BGR);
            using (Mat blurred = new Mat())
            {
                Cv2.GaussianBlur(croppedBgr, blurred, new OpenCvSharp.Size(0, 0), 1.5);
                Cv2.AddWeighted(croppedBgr, 1.5, blurred, -0.5, 0, croppedBgr);
            }

            using Mat croppedAlpha = new Mat();
            Cv2.ExtractChannel(cropped, croppedAlpha, 3);

            using Mat processed = new Mat();
            Cv2.CvtColor(croppedBgr, processed, ColorConversionCodes.BGR2RGBA);
            Cv2.MixChannels(new[] { croppedAlpha }, new[] { processed }, s_alphaMixMap);

            SaveDebugMat(processed, "09_cropped", fishIndex);
            Debug.Log($"[FinalizeOutput] {mat.Cols}×{mat.Rows} → {w}×{h}  crop=({x},{y})  fish{fishIndex}");
            return MatToTexture2D(processed);
        }

        return MatToTexture2D(mat);
    }

    // ── 세그멘테이션 방법 선택 dispatch ───────────────────────
    // 우선순위: AI(Sentis) > Contour(항상 직접 사용)
    // 이후 useInkLineClip=true 이면 2차 잉크 라인 클리핑 적용
    void DispatchSegmentation(Mat rgba, int fishIndex, int anchorPos = 0)
    {
        string dirLabel = anchorPos >= 0 && anchorPos < s_rotLabels.Length
            ? s_rotLabels[anchorPos] : anchorPos.ToString();
        Debug.Log($"[Segmentation] fish{fishIndex} — 방향: {dirLabel} (anchorPos={anchorPos})");

#if UNITY_SENTIS
        if (_sentisWorker != null)
            ApplyAISegmentation(rgba, fishIndex);
        else
#endif
        // 템플릿 존재 여부와 무관하게 항상 실제 그림 컨투어를 마스크로 사용
        // → 캡처마다 그림 크기가 달라도 컨투어가 그림에 정확히 맞춤
        ApplyContourSegmentation(rgba, fishIndex);

        // 2차 가공: 검정 외곽선 내부 정밀 클리핑
        if (useInkLineClip)
            ApplyInkLineClip(rgba, fishIndex);

        // 3차 가공: 방향별 마스크 위치 오프셋 (픽셀 단위 상하좌우 이동)
        if (maskOffsets == null || fishIndex >= maskOffsets.Length)
        {
            Debug.LogWarning($"[MaskOffset] 스킵 — maskOffsets.Length={maskOffsets?.Length ?? 0}, fishIndex={fishIndex}. " +
                             $"Inspector에서 배열 크기를 {fishIndex + 1} 이상으로 설정하세요.");
        }
        else
        {
            var off = maskOffsets[fishIndex].GetOffset(anchorPos);
            Debug.Log($"[MaskOffset] fish{fishIndex} anchorPos={anchorPos}({s_rotLabels[anchorPos]}) → offset=({off.x},{off.y})");
            if (off.x != 0 || off.y != 0)
                ShiftAlpha(rgba, off.x, off.y);
            else
                Debug.Log($"[MaskOffset] offset이 (0,0)이므로 ShiftAlpha 스킵. 올바른 방향에 값을 입력했는지 확인하세요.");
        }
    }

    // ── 알파 채널을 dx, dy 픽셀만큼 이동 ───────────────────────
    // dx > 0 = 오른쪽, dx < 0 = 왼쪽
    // dy > 0 = 아래쪽, dy < 0 = 위쪽
    static void ShiftAlpha(Mat rgba, int dx, int dy)
    {
        int W = rgba.Cols, H = rgba.Rows;

        // 현재 알파 추출
        using Mat src = new Mat();
        Cv2.ExtractChannel(rgba, src, 3);

        // 이동 행렬 적용 (WarpAffine)
        using Mat shifted = Mat.Zeros(H, W, MatType.CV_8UC1);
        using Mat transMat = new Mat(2, 3, MatType.CV_32F);
        transMat.Set<float>(0, 0, 1);  transMat.Set<float>(0, 1, 0);  transMat.Set<float>(0, 2, dx);
        transMat.Set<float>(1, 0, 0);  transMat.Set<float>(1, 1, 1);  transMat.Set<float>(1, 2, dy);
        Cv2.WarpAffine(src, shifted, transMat, new OpenCvSharp.Size(W, H),
                       InterpolationFlags.Nearest, BorderTypes.Constant, Scalar.Black);

        // 이동된 알파를 rgba에 적용
        Cv2.MixChannels(new[] { shifted }, new[] { rgba }, s_alphaMixMap);
        Debug.Log($"[MaskOffset] alpha shifted dx={dx} dy={dy}");
    }

    // ApplyInkLineClip  (2차 가공 — 잉크 외곽선 alpha 확장 + 경화)
    void ApplyInkLineClip(Mat rgba, int fishIndex)
    {
        int W = rgba.Cols, H = rgba.Rows;

        using Mat bgr = new Mat(); Cv2.CvtColor(rgba, bgr, ColorConversionCodes.RGBA2BGR);
        using Mat hsv = new Mat(); Cv2.CvtColor(bgr, hsv, ColorConversionCodes.BGR2HSV);
        using Mat vCh = new Mat(); Cv2.ExtractChannel(hsv, vCh, 2);

        // ① 잉크 마스크: V < inkInkThresh = 검정 외곽선
        using Mat darkMask = new Mat();
        Cv2.Threshold(vCh, darkMask, inkInkThresh, 255, ThresholdTypes.BinaryInv);

        // ② 현재 alpha 추출
        using Mat curAlpha = new Mat();
        Cv2.ExtractChannel(rgba, curAlpha, 3);
        int alphaArea = Cv2.CountNonZero(curAlpha);
        if (alphaArea < 100) return;

        SaveDebugMat(curAlpha, "ic_02_alpha_before", fishIndex);

        // ③ alpha 경계를 외곽 잉크까지 확장 + 경화
        // 선 굵기 절반(W/200 ≈ 7~15px)만큼 alpha 확장 → 검정 외곽선이 마스크 경계에 포함됨
        // 이전 Size(1,1) = 커널 크기 1×1 → 사실상 팽창 없음(버그) → 수정
        using Mat dilAlpha   = new Mat();
        using Mat outerInk   = new Mat();
        using Mat inkPresent = new Mat();
        int dilRadius = Mathf.Max(3, W / 200) | 1;
        using (var dilK = Cv2.GetStructuringElement(MorphShapes.Ellipse,
                   new OpenCvSharp.Size(dilRadius, dilRadius)))
            Cv2.Dilate(curAlpha, dilAlpha, dilK);       // alpha 외곽 선굵기 절반만큼 확장
        Cv2.BitwiseAnd(dilAlpha, darkMask, outerInk);   // 확장 영역 중 잉크(V<inkInkThresh)
        Cv2.BitwiseOr(curAlpha, outerInk, curAlpha);    // 외곽 잉크 → alpha=255 추가
        Cv2.BitwiseAnd(curAlpha, darkMask, inkPresent);
        Cv2.Threshold(inkPresent, inkPresent, 0, 255, ThresholdTypes.Binary);
        Cv2.BitwiseOr(curAlpha, inkPresent, curAlpha);  // alpha 내 잉크 전체 → alpha=255

        // ④ alpha 적용
        int finalArea = Cv2.CountNonZero(curAlpha);
        Cv2.MixChannels(new[] { curAlpha }, new[] { rgba }, s_alphaMixMap);

        Debug.Log($"[InkClip] 완료  alpha {alphaArea}→{finalArea}px  fish{fishIndex}");
        SaveDebugMat(rgba, "ic_04_clipped", fishIndex);
    }

    // ── [Contour Segmentation] 검정 외곽선 장벽 + 코너 FloodFill
    // 핵심 원리: 굵은 검정 외곽선만 장벽으로 추출 → 코너 FloodFill →
    //   외곽선 밖 볼펜 흔적은 배경으로 마킹 → 마스크에서 제외.
    void ApplyContourSegmentation(Mat rgba, int fishIndex)
    {
        int W = rgba.Cols, H = rgba.Rows;

        // 100% 처리: 다운스케일 없이 원본 해상도에서 직접 처리
        int sW = W, sH = H;
        Debug.Log($"[Contour] 입력:{W}×{H} 처리:100%  fish{fishIndex}");

        // ① 그레이스케일 (원본 해상도)
        using Mat sbgr = new Mat();
        Cv2.CvtColor(rgba, sbgr, ColorConversionCodes.RGBA2BGR);
        using Mat sgray = new Mat();
        Cv2.CvtColor(sbgr, sgray, ColorConversionCodes.BGR2GRAY);

        // ② 임계값: Otsu 자동 계산 후 200 캡 적용
        // Otsu: 종이 위치·조명 무관하게 최적 분리 (배경지 추정 불필요)
        int blurSz = Mathf.Max(3, (sW / 400) | 1);
        using Mat blurred = new Mat();
        Cv2.GaussianBlur(sgray, blurred, new OpenCvSharp.Size(blurSz, blurSz), 0);
        using Mat drawMask = new Mat();
        double otsuThresh = Cv2.Threshold(blurred, drawMask, 0, 255,
            ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);
        if (otsuThresh > 200)
        {
            // Otsu 임계값이 너무 높음 → 200으로 캡: 밝은 내부 픽셀도 포함
            Cv2.Threshold(blurred, drawMask, 200, 255, ThresholdTypes.BinaryInv);
            Debug.Log($"[Contour] Otsu={otsuThresh:F0} > 200 → 캡 적용: 임계=200  fish{fishIndex}");
        }
        else
        {
            Debug.Log($"[Contour] Otsu 임계값={otsuThresh:F0}  fish{fishIndex}");
        }
        SaveDebugMat(drawMask, "ct_01_darkmask", fishIndex);

        var inkBBox = new OpenCvSharp.Rect(sW / 8, sH / 8, sW * 6 / 8, sH * 6 / 8);
        {
            // ── ① 유채색 픽셀 마스크 생성 ──────────────────────────────
            using Mat hsv       = new Mat();
            using Mat satCh     = new Mat();
            using Mat colorMask = new Mat();
            Cv2.CvtColor(sbgr, hsv, ColorConversionCodes.BGR2HSV);
            Cv2.ExtractChannel(hsv, satCh, 1);                       // S 채널
            Cv2.Threshold(satCh, colorMask, 100, 255, ThresholdTypes.Binary);
            int dkSz = Mathf.Max(1, sW / 400) | 1;
            using (var dk = Cv2.GetStructuringElement(MorphShapes.Ellipse,
                                                       new OpenCvSharp.Size(dkSz, dkSz)))
                Cv2.Dilate(colorMask, colorMask, dk, iterations: 1);

            // ── ② drawMask에서 유채색 제거 → 검정 잉크 전용 bboxMask ──
            using Mat colorInv   = new Mat();
            using Mat bboxSource = new Mat();
            Cv2.BitwiseNot(colorMask, colorInv);
            Cv2.BitwiseAnd(drawMask, colorInv, bboxSource);
            SaveDebugMat(bboxSource, "ct_02_bboxsource", fishIndex);

            // drawMask 유채색 제거: 매우 어두운 픽셀(검정 잉크)은 채도와 무관하게 보존
            // → 외곽선이 채색 영역과 겹쳐도 FloodFill 장벽 픽셀이 유지되어 갭 발생 방지
            // veryDark: V < inkInkThresh/2 → 진짜 검정 잉크, 밝은 채색 필터에서 면제
            using Mat vCh        = new Mat();
            using Mat veryDark   = new Mat();
            using Mat drawFilter = new Mat();
            Cv2.ExtractChannel(hsv, vCh, 2);
            int veryDarkThresh = Mathf.Clamp(inkInkThresh / 2, 20, 60);
            Cv2.Threshold(vCh, veryDark, veryDarkThresh, 255, ThresholdTypes.BinaryInv);
            Cv2.BitwiseOr(colorInv, veryDark, drawFilter);  // 저채도 OR 매우어두움 → 보존
            Cv2.BitwiseAnd(drawMask, drawFilter, drawMask);
            SaveDebugMat(drawMask, "ct_01b_drawmask_clean", fishIndex);

            // ── ③ Close: 선 간격 연결 (Open 제거: 선폭≈5px이라 Open이 윤곽선 삭제 위험)
            using Mat bboxClosed = new Mat();
            using Mat bboxClean  = new Mat();
            using (var ck = CreateEllipseKernel(sW, 30))
                Cv2.MorphologyEx(bboxSource, bboxClosed, MorphTypes.Close, ck, iterations: 2);
            // FillHoles → 실선 링을 솔리드 blob으로 → 컨투어 면적이 명확해짐
            bboxClosed.CopyTo(bboxClean);
            ContourFillHoles(bboxClean, sW, sH);

            // ── ④ 컨투어 + centroid 대칭 bbox ─────────────────────────
            Cv2.FindContours(bboxClean, out Point[][] bboxContours, out _,
                RetrievalModes.External, ContourApproximationModes.ApproxSimple);
            double bboxMinArea = (double)sW * sH * 0.005;

            double sumX = 0, sumY = 0, sumArea = 0;
            int ux1 = sW, uy1 = sH, ux2 = 0, uy2 = 0;
            foreach (var c in bboxContours)
            {
                double a = Cv2.ContourArea(c);
                if (a < bboxMinArea) continue;
                var m = Cv2.Moments(c);
                if (m.M00 > 0) { sumX += m.M10; sumY += m.M01; sumArea += m.M00; }
                var r = Cv2.BoundingRect(c);
                ux1 = Mathf.Min(ux1, r.X);           uy1 = Mathf.Min(uy1, r.Y);
                ux2 = Mathf.Max(ux2, r.X + r.Width); uy2 = Mathf.Max(uy2, r.Y + r.Height);
            }

            if (sumArea > 0 && ux2 > ux1)
            {
                // 실제 union bounds 그대로 사용: 비대칭 구조(꼬리·지느러미)도 잘리지 않음
                // 이전 centroid 대칭 방식은 무게중심이 bbox 중앙과 다를 때 한쪽을 잘라냄
                inkBBox = new OpenCvSharp.Rect(
                    ux1,
                    uy1,
                    Mathf.Min(sW - ux1, ux2 - ux1),
                    Mathf.Min(sH - uy1, uy2 - uy1));
                int cx = (int)(sumX / sumArea);
                int cy = (int)(sumY / sumArea);
                Debug.Log($"[Contour] inkBBox(union bounds) c=({cx},{cy}) → ({inkBBox.X},{inkBBox.Y}) {inkBBox.Width}×{inkBBox.Height}  fish{fishIndex}");
            }
            else
            {
                // 폴백: 색상 필터 없이 raw drawMask 전체 픽셀
                using Mat inkPts = new Mat();
                Cv2.FindNonZero(drawMask, inkPts);
                if (inkPts != null && !inkPts.Empty())
                    inkBBox = Cv2.BoundingRect(inkPts);
                Debug.Log($"[Contour] inkBBox(raw fallback) → ({inkBBox.X},{inkBBox.Y}) {inkBBox.Width}×{inkBBox.Height}  fish{fishIndex}");
            }
        }

        // ── inkBBox 기반 동적 커널 크기 계산 ──────────────────────────────
        // sW(전체 캔버스) 대신 inkBBox(실제 그림 범위)를 기준으로 커널 산출
        // → 카메라 거리·그림 크기가 달라도 선 굵기 대비 커널 비율이 일정하게 유지됨
        // 예) 그림이 작을 때(inkDim=600): 커널 작아짐 → 과도한 팽창 방지
        //     그림이 클 때(inkDim=1800): 커널 커짐  → 넓은 간격도 연결 가능
        // sW/2 를 하한으로 두어 inkBBox 미검출 시 기존 동작 유지
        int inkDim = inkBBox.Width > 0 && inkBBox.Height > 0
            ? Mathf.Max(inkBBox.Width, inkBBox.Height)
            : sW / 2;
        Debug.Log($"[Contour] inkDim={inkDim} (inkBBox:{inkBBox.Width}×{inkBBox.Height} sW:{sW})  fish{fishIndex}");

        // ④-pre: Large Close 전 inkBBox 클립
        // drawMask에 inkBBox 바깥의 dark 픽셀(조명 불균일·그림자·종이 경계 외 영역)이 포함되면
        // Large Close 후 blob이 캔버스 전체를 덮어 biggestContour가 이미지 전체 크기가 됨
        // → inkBBox 영역(+여유 패딩)으로 처리 범위를 한정
        {
            int inkPad = Mathf.Max(30, inkDim / 8);
            int cx = Mathf.Max(0, inkBBox.X - inkPad);
            int cy = Mathf.Max(0, inkBBox.Y - inkPad);
            int cw = Mathf.Min(sW - cx, inkBBox.Width  + inkPad * 2);
            int ch = Mathf.Min(sH - cy, inkBBox.Height + inkPad * 2);
            if (cw > 0 && ch > 0)
            {
                using Mat bboxClipMask = Mat.Zeros(sH, sW, MatType.CV_8UC1);
                using (Mat roi = new Mat(bboxClipMask,
                       new OpenCvSharp.Rect(cx, cy, cw, ch)))
                    roi.SetTo(Scalar.White);
                Cv2.BitwiseAnd(drawMask, bboxClipMask, drawMask);
                SaveDebugMat(drawMask, "ct_01d_drawmask_inkclip", fishIndex);
            }
        }

        // ④ Large Close: 외곽선 간격 연결 (inkDim/40 × 2회)
        using Mat closedBlob = new Mat();
        using (var ck = CreateEllipseKernel(inkDim, 10))
            Cv2.MorphologyEx(drawMask, closedBlob, MorphTypes.Close, ck, iterations: 2);
        SaveDebugMat(closedBlob, "ct_02_closed", fishIndex);

        // ⑤ ContourFillHoles: 연결된 외곽선 내부를 솔리드 blob으로 채움
        // Open 대신 FillHoles 사용: Open은 연결된 선 blob 자체를 제거할 위험 있음
        ContourFillHoles(closedBlob, sW, sH);
        SaveDebugMat(closedBlob, "ct_03_filled", fishIndex);

        // ⑥ 유효 컨투어 수집 (closedBlob에서 직접 — 솔리드 blob 기준)
        Cv2.FindContours(closedBlob, out Point[][] contours, out _,
            RetrievalModes.External, ContourApproximationModes.ApproxNone);
        if (contours.Length == 0)
        {
            Debug.LogWarning("[Contour] 컨투어 없음 → closedBlob 직접 마스크로 대체");
            ApplyClosedBlobMask(rgba, closedBlob, inkBBox, W, H, fishIndex);
            return;
        }

        double maxAllowed = (double)sW * sH * 0.90; // 90% 이상 = 캔버스 전체 루프 → 제외
        double minArea    = (double)sW * sH * 0.005; // 0.5% 미만 잡음 제거
        var validContours = new System.Collections.Generic.List<Point[]>();
        double totalArea  = 0;
        foreach (var c in contours)
        {
            double a = Cv2.ContourArea(c);
            if (a >= minArea && a <= maxAllowed) { validContours.Add(c); totalArea += a; }
        }
        if (validContours.Count == 0)
        {
            Debug.LogWarning("[Contour] 최소 면적 컨투어 없음 → 기준 완화 후 재시도");
            double relaxedArea = (double)sW * sH * 0.001; // 0.1%로 완화
            foreach (var c in contours)
            {
                double a = Cv2.ContourArea(c);
                if (a >= relaxedArea && a <= maxAllowed) { validContours.Add(c); totalArea += a; }
            }
            if (validContours.Count == 0)
            {
                Debug.LogWarning("[Contour] 완화 후에도 컨투어 없음 → closedBlob 직접 마스크로 대체");
                ApplyClosedBlobMask(rgba, closedBlob, inkBBox, W, H, fishIndex);
                return;
            }
            Debug.Log($"[Contour] 완화 기준으로 컨투어 {validContours.Count}개 확보  fish{fishIndex}");
        }
        Debug.Log($"[Contour] 유효 컨투어 {validContours.Count}개  totalArea={totalArea:F0}px²  fish{fishIndex}");

        // ⑦ 최대 컨투어 선택 → DrawContours 완전 채우기 (내부 구멍 원천 제거)
        Point[] biggestContour = validContours[0];
        double biggestArea = Cv2.ContourArea(validContours[0]);
        for (int ci = 1; ci < validContours.Count; ci++)
        {
            double a = Cv2.ContourArea(validContours[ci]);
            if (a > biggestArea) { biggestArea = a; biggestContour = validContours[ci]; }
        }
        Debug.Log($"[Contour] 최대 컨투어 면적={biggestArea:F0}px²  fish{fishIndex}");
        using Mat alphaMask = Mat.Zeros(sH, sW, MatType.CV_8UC1);
        Cv2.DrawContours(alphaMask, new[] { biggestContour }, 0, Scalar.White, thickness: -1);

        // ⑦-b Close/Erode 전 inkBBox 선클립: alphaMask를 실제 그림 범위로 제한
        // 이렇게 해야 Erode가 캔버스 전체가 아닌 실제 거북이 경계에서 작동
        {
            int preClipPad = Mathf.Max(10, inkDim / 15);
            int pcX = Mathf.Max(0, inkBBox.X - preClipPad);
            int pcY = Mathf.Max(0, inkBBox.Y - preClipPad);
            int pcW = Mathf.Min(sW - pcX, inkBBox.Width  + preClipPad * 2);
            int pcH = Mathf.Min(sH - pcY, inkBBox.Height + preClipPad * 2);
            if (pcW > 0 && pcH > 0)
            {
                using Mat preClipMask = Mat.Zeros(sH, sW, MatType.CV_8UC1);
                using (Mat roi = new Mat(preClipMask,
                       new OpenCvSharp.Rect(pcX, pcY, pcW, pcH)))
                    roi.SetTo(Scalar.White);
                Cv2.BitwiseAnd(alphaMask, preClipMask, alphaMask);
            }
        }
        SaveDebugMat(alphaMask, "ct_04_fillpoly", fishIndex);

        // ⑨ 경계 스무딩 Close: 오목한 윤곽 연결
        using (var concaveK = CreateEllipseKernel(inkDim, 100))
            Cv2.MorphologyEx(alphaMask, alphaMask, MorphTypes.Close, concaveK, iterations: 1);
        SaveDebugMat(alphaMask, "ct_06_concave", fishIndex);

        // ⑩ 마스크 수축(Erode): 검정 외곽선 안으로 파고들기
        using (var erodeK = CreateEllipseKernel(inkDim, 200))
            Cv2.Erode(alphaMask, alphaMask, erodeK, iterations: 1);
        SaveDebugMat(alphaMask, "ct_07_eroded", fishIndex);

        // sW=W, sH=H이므로 alphaMask가 이미 W×H — 업스케일 불필요
        Cv2.Threshold(alphaMask, alphaMask, 127, 255, ThresholdTypes.Binary);

        // ⑫-b ★ 잉크 경계 클립: 실제 잉크 범위(inkBBox) 바깥 알파 강제 제거
        {
            int inkPad = Mathf.Max(5, W / 60);
            int clipX = Mathf.Max(0,     inkBBox.X      - inkPad);
            int clipY = Mathf.Max(0,     inkBBox.Y      - inkPad);
            int clipW = Mathf.Min(W - clipX, inkBBox.Width  + inkPad * 2);
            int clipH = Mathf.Min(H - clipY, inkBBox.Height + inkPad * 2);

            using Mat clipMask = Mat.Zeros(H, W, MatType.CV_8UC1);
            using (Mat roi = new Mat(clipMask, new OpenCvSharp.Rect(clipX, clipY, clipW, clipH)))
                roi.SetTo(Scalar.White);

            Cv2.BitwiseAnd(alphaMask, clipMask, alphaMask);
            Debug.Log($"[Contour] inkClip=({clipX},{clipY}) {clipW}×{clipH}  pad={inkPad}  fish{fishIndex}");
        }

        Cv2.MixChannels(new[] { alphaMask }, new[] { rgba }, s_alphaMixMap);
        Debug.Log($"[Contour] 완료: fish{fishIndex}  totalArea={totalArea:F0}px²  contours={validContours.Count}");
        SaveDebugMat(rgba, "ct_10_final", fishIndex);
    }

    // ── 컨투어 검출 실패 시: closedBlob + FillHoles → 직접 마스크 적용 ──
    // 템플릿 없이 실제 잉크 픽셀 기반으로 마스크 생성 → 크기 변화에 자동 적응
    void ApplyClosedBlobMask(Mat rgba, Mat closedBlob,
                             OpenCvSharp.Rect inkBBox, int W, int H, int fishIndex)
    {
        using Mat fallback = closedBlob.Clone();

        // 내부 구멍 채우기: Dilate 봉쇄 → FillHoles → Erode 복원
        {
            int dim    = Mathf.Max(inkBBox.Width, inkBBox.Height);
            int sealPx = Mathf.Max(3, dim / 200);
            using var sealK = Cv2.GetStructuringElement(MorphShapes.Ellipse,
                new OpenCvSharp.Size(sealPx * 2 + 1, sealPx * 2 + 1));
            using Mat fbSeal = new Mat();
            Cv2.Dilate(fallback, fbSeal, sealK, iterations: 1);
            ContourFillHoles(fbSeal, W, H);
            Cv2.Erode(fbSeal, fallback, sealK, iterations: 1);
        }
        Cv2.Threshold(fallback, fallback, 127, 255, ThresholdTypes.Binary);

        // inkBBox 클립: 잉크 범위 바깥 제거
        if (inkBBox.Width > 0 && inkBBox.Height > 0)
        {
            int pad = Mathf.Max(5, W / 60);
            int cx = Mathf.Max(0, inkBBox.X - pad);
            int cy = Mathf.Max(0, inkBBox.Y - pad);
            int cw = Mathf.Min(W - cx, inkBBox.Width  + pad * 2);
            int ch = Mathf.Min(H - cy, inkBBox.Height + pad * 2);
            using Mat clipMask = Mat.Zeros(H, W, MatType.CV_8UC1);
            using (Mat roi = new Mat(clipMask, new OpenCvSharp.Rect(cx, cy, cw, ch)))
                roi.SetTo(Scalar.White);
            Cv2.BitwiseAnd(fallback, clipMask, fallback);
        }

        Cv2.MixChannels(new[] { fallback }, new[] { rgba }, s_alphaMixMap);
        Debug.Log($"[Contour] closedBlob 직접 마스크 적용 완료  fish{fishIndex}");
        SaveDebugMat(rgba, "ct_10_final", fishIndex);
    }

    static void ContourFillHoles(Mat binaryMask, int W, int H)
    {
        using Mat inv = new Mat();
        Cv2.BitwiseNot(binaryMask, inv);

        using Mat filled = inv.Clone();
        using Mat ffMask = Mat.Zeros(inv.Rows + 2, inv.Cols + 2, MatType.CV_8UC1);
        // 4코너에서 FloodFill → 외부 배경을 검정(0)으로 제거
        // s_holeSeeds: static 배열 재사용으로 매 호출 배열 할당 방지
        s_holeSeeds[0] = new Point(0,           0          );
        s_holeSeeds[1] = new Point(inv.Cols-1,  0          );
        s_holeSeeds[2] = new Point(0,           inv.Rows-1 );
        s_holeSeeds[3] = new Point(inv.Cols-1,  inv.Rows-1 );
        foreach (var seed in s_holeSeeds)
        {
            if (filled.At<byte>(seed.Y, seed.X) == 255)
                Cv2.FloodFill(filled, ffMask, seed, Scalar.Black);
        }

        if (Cv2.CountNonZero(filled) < (long)W * H * 0.5)
            Cv2.BitwiseOr(binaryMask, filled, binaryMask);
        else
            Debug.LogWarning("[ContourFillHoles] 구멍 면적 과다 → FillHoles 건너뜀 (all-white 방지)");
    }

    // ── 해상도 비례 타원형 구조 요소 생성 ────────────────────
    static Mat CreateEllipseKernel(int dimension, int divisor)
    {
        int radius = Mathf.Max(1, dimension / divisor);
        return Cv2.GetStructuringElement(
            MorphShapes.Ellipse,
            new OpenCvSharp.Size(radius * 2 + 1, radius * 2 + 1));
    }

#if UNITY_SENTIS
    void ApplyAISegmentation(Mat rgba, int fishIndex)
    {
        int W = rgba.Cols, H = rgba.Rows;
        int sz = aiInputSize;
        Debug.Log($"[AISeg] ① 입력: {W}×{H}  modelInput={sz}×{sz}  fish{fishIndex}");

        float[] inputData = AISeg_Preprocess(rgba, sz);

        using var inputTensor = new Tensor<float>(new TensorShape(1, 3, sz, sz), inputData);
        _sentisWorker.Schedule(inputTensor);

        using var rawTensor = _sentisWorker.PeekOutput() as Tensor<float>;
        if (rawTensor == null)
        {
            Debug.LogError("[AISeg] 출력 텐서 취득 실패 → 처리 중단");
            return;
        }
        // Sentis 2.x: GPU 텐서 → CPU 복사본으로 변환해야 인덱서 접근 가능
        using var salTensor = rawTensor.ReadbackAndClone();
        Debug.Log($"[AISeg] ② 추론 완료  shape={salTensor.shape}");

        using Mat salMat  = AISeg_SaliencyToMat(salTensor, sz, sz);
        SaveDebugMat(salMat, "02_saliency_raw", fishIndex);

        using Mat resized = new Mat();
        Cv2.Resize(salMat, resized, new OpenCvSharp.Size(W, H));


        using Mat alphaMask = new Mat();
        // ③ 이진화: threshold가 너무 높아 픽셀이 거의 없으면 자동으로 낮춤 (Otsu 폴백)
        Cv2.Threshold(resized, alphaMask, (int)(aiThreshold * 255), 255, ThresholdTypes.Binary);
        int pxAfterThresh = Cv2.CountNonZero(alphaMask);
        Debug.Log($"[AISeg] ③ 이진화(threshold={aiThreshold:F2}): " +
                  $"마스크={pxAfterThresh}px / {W * H}px");
        // 픽셀이 전체의 1% 미만이면 threshold가 너무 높은 것 → Otsu 자동 임계값으로 재시도
        if (pxAfterThresh < W * H * 0.01)
        {
            double otsuThresh = Cv2.Threshold(resized, alphaMask, 0, 255,
                ThresholdTypes.Binary | ThresholdTypes.Otsu);
            pxAfterThresh = Cv2.CountNonZero(alphaMask);
            Debug.LogWarning($"[AISeg] ③ threshold 너무 높음 → Otsu 폴백({otsuThresh/255.0:F2}): {pxAfterThresh}px");
        }
        SaveDebugMat(alphaMask, "03_binary_thresh", fishIndex);

        // ④ 마스크 침식: AI가 그림보다 크게 잡는 문제 보정
        //    단, 침식 후 픽셀이 0이 되면 침식 자체를 건너뜀 (FillHoles가 전체 캔버스를 채우는 버그 방지)
        if (aiMaskErodePx > 0)
        {
            // 침식 크기를 캔버스 단변의 5% 이하로 안전 제한 (너무 큰 값 방지)
            int safeErodePx = Mathf.Min(aiMaskErodePx, Mathf.Min(W, H) / 20);
            int eks = safeErodePx * 2 + 1;
            using Mat ek = Cv2.GetStructuringElement(MorphShapes.Ellipse,
                new OpenCvSharp.Size(eks, eks));
            using Mat eroded = new Mat();
            Cv2.Erode(alphaMask, eroded, ek);
            int pxAfterErode = Cv2.CountNonZero(eroded);
            if (pxAfterErode > 0)
            {
                eroded.CopyTo(alphaMask);
                Debug.Log($"[AISeg] ④ 침식({safeErodePx}px, 원본요청={aiMaskErodePx}px) 후: {pxAfterErode}px");
            }
            else
            {
                Debug.LogWarning($"[AISeg] ④ 침식({safeErodePx}px) 시 0px → 침식 건너뜀 (원본 {pxAfterThresh}px 유지)");
            }
            SaveDebugMat(alphaMask, "04_after_erode", fishIndex);
        }

        // ⑤ 꼬리·지느러미 내부 구멍 채우기 (FloodFill 역방향)
        //    입력이 0px이면 FillHoles가 전체 캔버스를 채워버리므로 반드시 guard 필요
        if (aiHoleFill && Cv2.CountNonZero(alphaMask) > 0)
        {
            AISeg_FillHoles(alphaMask);
            Debug.Log($"[AISeg] ⑤ 구멍채우기 후: {Cv2.CountNonZero(alphaMask)}px");
            SaveDebugMat(alphaMask, "05_after_holefill", fishIndex);
        }
        else if (aiHoleFill)
        {
            Debug.LogWarning($"[AISeg] ⑤ 마스크 0px → 구멍채우기 건너뜀 (전체 캔버스 fill 방지)");
        }

        // ⑥ FishMaskApplier AND 제약: 외부 그림(붉은색 등) 차단 (항상 적용)
        Mat cachedMask = (_fishApplierMasks != null && fishIndex < _fishApplierMasks.Length)
                       ? _fishApplierMasks[fishIndex] : null;
        if (cachedMask != null)
        {
            Debug.Log($"[AISeg] ⑥ FishMask 리사이즈: 파일={cachedMask.Cols}×{cachedMask.Rows} → 캡처={W}×{H}px");

            using Mat resizedMask = new Mat();
            Cv2.Resize(cachedMask, resizedMask, new OpenCvSharp.Size(W, H),
                       0, 0, InterpolationFlags.Lanczos4);

            // 리사이즈된 마스크를 W×H 캔버스 중앙에 배치 후 AND
            using Mat centered = CenterPlaceMask(resizedMask, W, H);
            Cv2.BitwiseAnd(alphaMask, centered, alphaMask);
            Debug.Log($"[AISeg] ⑥ FishMaskApplier AND 후: {Cv2.CountNonZero(alphaMask)}px");
            SaveDebugMat(alphaMask, "06_after_and", fishIndex);

            // ⑥-a AND 이후 구멍 채우기 (BitwiseNot 없는 정확한 구현)
            // 문제: ⑤ AISeg_FillHoles(BitwiseNot 방식)가 enclosed hole(눈 등)을 black으로 남김
            //       → AND(black, template_white) = black → 구멍이 AND 이후에도 살아남음
            //       → ShiftAlpha 후 구멍 위치가 offset에 따라 그림 몸통 위로 이동 → 보임
            // 해결: BitwiseNot 없이 filled를 직접 OR → 배경 오염 없이 구멍만 흰색으로 채움
            if (Cv2.CountNonZero(alphaMask) > 0)
            {
                FillHolesAfterAnd(alphaMask);
                Debug.Log($"[AISeg] ⑥-a AND 후 구멍채우기: {Cv2.CountNonZero(alphaMask)}px");
                SaveDebugMat(alphaMask, "06a_after_and_holefill", fishIndex);
            }

            // ⑥-b AND 이후 침식: AND가 템플릿 크기를 그대로 반환하는 경우를 보정
            if (aiMaskErodePx > 0)
            {
                int safeErodePx = Mathf.Min(aiMaskErodePx, Mathf.Min(W, H) / 20);
                int eks = safeErodePx * 2 + 1;
                using Mat postK = Cv2.GetStructuringElement(MorphShapes.Ellipse,
                    new OpenCvSharp.Size(eks, eks));
                using Mat postEroded = new Mat();
                Cv2.Erode(alphaMask, postEroded, postK);
                int pxAfterPostErode = Cv2.CountNonZero(postEroded);
                if (pxAfterPostErode > 0)
                {
                    postEroded.CopyTo(alphaMask);
                    Debug.Log($"[AISeg] ⑥-b AND 후 침식({safeErodePx}px) 후: {pxAfterPostErode}px");
                }
                else
                {
                    Debug.LogWarning($"[AISeg] ⑥-b AND 후 침식({safeErodePx}px) 시 0px → 침식 건너뜀 (AND 결과 유지)");
                }
                SaveDebugMat(alphaMask, "06b_after_and_erode", fishIndex);
            }
        }
        else
        {
            Debug.LogWarning($"[AISeg] ⑥ FishMaskApplier fish{fishIndex} 없음 → AND 건너뜀");
        }

        // ⑦-pre: 실제 잉크(검정 외곽선) 기반 inkBBox 감지 → alphaMask 외곽 클리핑
        // 목적: FishMaskApplier 템플릿은 고정 크기(실제 그림보다 항상 큼) →
        //       실제 잉크 픽셀의 BoundingRect 로 alphaMask를 한정해야 여백 제거 가능
        {
            using Mat aisgBgr = new Mat();
            Cv2.CvtColor(rgba, aisgBgr, ColorConversionCodes.RGBA2BGR);
            using Mat aisgHsv = new Mat();
            Cv2.CvtColor(aisgBgr, aisgHsv, ColorConversionCodes.BGR2HSV);

            // V 채널: 어두운 픽셀(검정 잉크 외곽선)
            using Mat aisgV = new Mat();
            Cv2.ExtractChannel(aisgHsv, aisgV, 2);
            using Mat aisgDark = new Mat();
            Cv2.Threshold(aisgV, aisgDark, inkInkThresh, 255, ThresholdTypes.BinaryInv);

            // S 채널: 유채색 픽셀 제거 (채색 영역이 어둡게 잡히는 오인식 방지)
            using Mat aisgS = new Mat();
            Cv2.ExtractChannel(aisgHsv, aisgS, 1);
            using Mat aisgColor = new Mat();
            Cv2.Threshold(aisgS, aisgColor, 80, 255, ThresholdTypes.Binary);
            using Mat aisgColorInv = new Mat();
            Cv2.BitwiseNot(aisgColor, aisgColorInv);
            Cv2.BitwiseAnd(aisgDark, aisgColorInv, aisgDark); // 저채도 어두운 픽셀 = 잉크만

            // 잉크 픽셀 BoundingRect
            using Mat inkPtsMat = new Mat();
            Cv2.FindNonZero(aisgDark, inkPtsMat);
            if (inkPtsMat != null && !inkPtsMat.Empty())
            {
                var rawInkBBox = Cv2.BoundingRect(inkPtsMat);
                // 여유 패딩: inkBBox의 10% 또는 최소 30px
                int ibPad = Mathf.Max(30, Mathf.Max(rawInkBBox.Width, rawInkBBox.Height) / 10);
                int ibX = Mathf.Max(0, rawInkBBox.X - ibPad);
                int ibY = Mathf.Max(0, rawInkBBox.Y - ibPad);
                int ibW = Mathf.Min(W - ibX, rawInkBBox.Width  + ibPad * 2);
                int ibH = Mathf.Min(H - ibY, rawInkBBox.Height + ibPad * 2);
                Debug.Log($"[AISeg] ⑦-pre inkBBox: raw=({rawInkBBox.X},{rawInkBBox.Y}) " +
                          $"{rawInkBBox.Width}×{rawInkBBox.Height}  pad={ibPad}  " +
                          $"clip=({ibX},{ibY}) {ibW}×{ibH}  fish{fishIndex}");

                // alphaMask를 inkBBox 바깥에서 0으로 클리핑
                using Mat inkClipMask = Mat.Zeros(H, W, MatType.CV_8UC1);
                using (Mat inkRoi = new Mat(inkClipMask, new OpenCvSharp.Rect(ibX, ibY, ibW, ibH)))
                    inkRoi.SetTo(Scalar.White);
                Cv2.BitwiseAnd(alphaMask, inkClipMask, alphaMask);
                SaveDebugMat(alphaMask, "06b_after_inkclip", fishIndex);
            }
            else
            {
                Debug.LogWarning($"[AISeg] ⑦-pre 잉크 픽셀 없음 → inkBBox 클리핑 건너뜀  fish{fishIndex}");
            }
        }

        // ⑧ 알파 경계 anti-alias: 5×5 GaussianBlur로 hard edge → 부드러운 전환
        Cv2.GaussianBlur(alphaMask, alphaMask, new OpenCvSharp.Size(5, 5), 1.5);

        Cv2.MixChannels(new[] { alphaMask }, new[] { rgba }, s_alphaMixMap);
        Debug.Log($"[AISeg] ⑨ 완료: {W}×{H}  Fish{fishIndex}");
    }

    // ── 내부 구멍 채우기 ─────────────────────────────────────
    // 이진 마스크에서 외부와 연결되지 않은 검정 영역(구멍)을 흰색으로 채움
    // 원리: 반전 → 전체 테두리 FloodFill(배경 마킹) → 재반전 → OR 병합
    // 개선: 4코너 씨드 → 4변 전체 테두리 씨드
    //       물고기 방향/꼬리 위치에 따라 코너 FloodFill이 배경을 누락하는 문제 해결
    static void AISeg_FillHoles(Mat binaryMask)
    {
        using Mat inv = new Mat();
        Cv2.BitwiseNot(binaryMask, inv);

        using Mat filled = inv.Clone();
        using Mat ffMask = Mat.Zeros(inv.Rows + 2, inv.Cols + 2, MatType.CV_8UC1);

        // 전체 테두리(4변)에서 흰 픽셀마다 FloodFill → 외부 배경을 검정(0)으로 마킹
        // 코너 4개만 씨드로 쓸 경우 물고기 몸통/꼬리가 엣지에 닿으면
        // 고립된 배경 영역을 내부 구멍으로 오인하는 방향 의존 버그를 방지
        int cols = inv.Cols, rows = inv.Rows;
        for (int x = 0; x < cols; x++)
        {
            if (filled.At<byte>(0, x) == 255)
                Cv2.FloodFill(filled, ffMask, new Point(x, 0), Scalar.Black);
            if (filled.At<byte>(rows - 1, x) == 255)
                Cv2.FloodFill(filled, ffMask, new Point(x, rows - 1), Scalar.Black);
        }
        for (int y = 0; y < rows; y++)
        {
            if (filled.At<byte>(y, 0) == 255)
                Cv2.FloodFill(filled, ffMask, new Point(0, y), Scalar.Black);
            if (filled.At<byte>(y, cols - 1) == 255)
                Cv2.FloodFill(filled, ffMask, new Point(cols - 1, y), Scalar.Black);
        }

        // 남은 흰 픽셀 = 내부 구멍
        Cv2.BitwiseNot(filled, filled);
        Cv2.BitwiseOr(binaryMask, filled, binaryMask);
    }

    // ── AND 이후 구멍 채우기 (BitwiseNot 없는 정확한 구현) ────────
    // AISeg_FillHoles와 달리 BitwiseNot을 사용하지 않으므로:
    //   - 배경이 흰색으로 오염되지 않음 (배경 = 검정 유지)
    //   - enclosed hole만 정확히 흰색으로 채움
    // AND(all_white_minus_eye, template)로 생긴 눈 구멍 등 처리용
    static void FillHolesAfterAnd(Mat binaryMask)
    {
        using Mat inv = new Mat();
        Cv2.BitwiseNot(binaryMask, inv);

        using Mat filled = inv.Clone();
        using Mat ffMask = Mat.Zeros(inv.Rows + 2, inv.Cols + 2, MatType.CV_8UC1);
        int cols = inv.Cols, rows = inv.Rows;
        for (int x = 0; x < cols; x++)
        {
            if (filled.At<byte>(0, x) == 255)
                Cv2.FloodFill(filled, ffMask, new Point(x, 0), Scalar.Black);
            if (filled.At<byte>(rows - 1, x) == 255)
                Cv2.FloodFill(filled, ffMask, new Point(x, rows - 1), Scalar.Black);
        }
        for (int y = 0; y < rows; y++)
        {
            if (filled.At<byte>(y, 0) == 255)
                Cv2.FloodFill(filled, ffMask, new Point(0, y), Scalar.Black);
            if (filled.At<byte>(y, cols - 1) == 255)
                Cv2.FloodFill(filled, ffMask, new Point(cols - 1, y), Scalar.Black);
        }
        // filled = [fish=black, 배경=black, 구멍=white]
        // BitwiseNot 없이 직접 OR → 구멍만 흰색, 배경 오염 없음
        Cv2.BitwiseOr(binaryMask, filled, binaryMask);
    }

    float[] AISeg_Preprocess(Mat rgba, int sz)
    {
        using Mat rgb = new Mat();
        Cv2.CvtColor(rgba, rgb, ColorConversionCodes.RGBA2RGB);
        using Mat resized = new Mat();
        Cv2.Resize(rgb, resized, new OpenCvSharp.Size(sz, sz), interpolation: InterpolationFlags.Linear);

        // 단일 Marshal.Copy로 전체 픽셀 복사 — 픽셀당 interop 제거
        int totalBytes = sz * sz * 3;
        if (_aisegRawBuf == null || _aisegRawBuf.Length < totalBytes)
            _aisegRawBuf = new byte[totalBytes];
        Marshal.Copy(resized.Data, _aisegRawBuf, 0, totalBytes);

        int plane = sz * sz;
        int dataLen = 3 * plane;
        if (_aisegFloatBuf == null || _aisegFloatBuf.Length < dataLen)
            _aisegFloatBuf = new float[dataLen];
        float[] data = _aisegFloatBuf;

        float[] mean = { 0.485f, 0.456f, 0.406f };
        float[] std  = { 0.229f, 0.224f, 0.225f };

        for (int i = 0; i < plane; i++)
        {
            int src = i * 3;
            data[          i] = (_aisegRawBuf[src    ] / 255f - mean[0]) / std[0]; // R
            data[plane   + i] = (_aisegRawBuf[src + 1] / 255f - mean[1]) / std[1]; // G
            data[plane*2 + i] = (_aisegRawBuf[src + 2] / 255f - mean[2]) / std[2]; // B
        }
        return data;
    }

    Mat AISeg_SaliencyToMat(Tensor<float> tensor, int W, int H)
    {
        float cx = W * 0.5f, cy = H * 0.5f;
        float sigX = aiCenterSigma * W, sigY = aiCenterSigma * H;
        bool  useGauss = aiCenterSigma > 0.01f;

        int pixCount = H * W;
        if (_aisegPixelBuf == null || _aisegPixelBuf.Length < pixCount)
            _aisegPixelBuf = new byte[pixCount];
        byte[] pixels = _aisegPixelBuf;
        float salMin = float.MaxValue, salMax = float.MinValue;

        for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
        {
            float sal = tensor[0, 0, y, x];
            salMin = Mathf.Min(salMin, sal);
            salMax = Mathf.Max(salMax, sal);

            if (useGauss)
            {
                float dx = x - cx, dy = y - cy;
                float gw = Mathf.Exp(-(dx * dx / (2f * sigX * sigX)
                                    + dy * dy / (2f * sigY * sigY)));
                sal *= gw;
            }
            pixels[y * W + x] = (byte)Mathf.Clamp(sal * 255f, 0f, 255f);
        }

        Mat mat = new Mat(H, W, MatType.CV_8UC1);
        Marshal.Copy(pixels, 0, mat.Data, pixels.Length);
        Debug.Log($"[AISeg] 살리언시 범위: {salMin:F3}~{salMax:F3}  가우시안={useGauss}(σ={aiCenterSigma:F2})");
        return mat;
    }
#endif

    // ── FishMaskApplier 텍스처 → 1채널 마스크 Mat ────────────
    // 알파 채널이 있으면 알파 사용, 없으면 grayscale 반전(흰=마스크)
    static Mat ApplierTextureToMask(Texture2D tex)
    {
        Color32[] pixels = tex.GetPixels32();
        byte[] data = new byte[pixels.Length];

        // 알파 채널 판별: 샘플 픽셀 중 alpha<250 가 있으면 알파 채널 사용
        bool hasAlpha = false;
        for (int i = 0; i < Mathf.Min(pixels.Length, 1000); i++)
            if (pixels[i].a < 250) { hasAlpha = true; break; }

        if (hasAlpha)
        {
            for (int i = 0; i < pixels.Length; i++) data[i] = pixels[i].a;
        }
        else
        {
            // 알파 없으면 grayscale 직접 사용 (흰=물고기 영역, 검정=배경)
            for (int i = 0; i < pixels.Length; i++)
                data[i] = (byte)((pixels[i].r + pixels[i].g + pixels[i].b) / 3);
        }

        Mat mat = new Mat(tex.height, tex.width, MatType.CV_8UC1);
        Marshal.Copy(data, 0, mat.Data, data.Length);
        Cv2.Flip(mat, mat, FlipMode.X);
        Cv2.Threshold(mat, mat, 128, 255, ThresholdTypes.Binary);
        return mat;
    }

    // ── 마스크 중앙 배치 ──────────────────────────────────────
    static Mat CenterPlaceMask(Mat src, int canvasW, int canvasH)
    {
        Mat canvas = Mat.Zeros(canvasH, canvasW, MatType.CV_8UC1);

        int srcW = src.Cols, srcH = src.Rows;

        // 캔버스 위 붙여넣을 영역 (canvas 좌표)
        int dstX = (canvasW - srcW) / 2;
        int dstY = (canvasH - srcH) / 2;

        // src 중 실제 복사할 영역 (src 좌표)
        int srcStartX = 0, srcStartY = 0;

        // src가 캔버스보다 크면: src를 크롭
        if (dstX < 0) { srcStartX = -dstX; dstX = 0; }
        if (dstY < 0) { srcStartY = -dstY; dstY = 0; }

        int copyW = Mathf.Min(srcW - srcStartX, canvasW - dstX);
        int copyH = Mathf.Min(srcH - srcStartY, canvasH - dstY);

        if (copyW <= 0 || copyH <= 0) return canvas;

        var srcRoi    = new OpenCvSharp.Rect(srcStartX, srcStartY, copyW, copyH);
        var canvasRoi = new OpenCvSharp.Rect(dstX,      dstY,      copyW, copyH);

        using Mat srcRegion    = new Mat(src,    srcRoi);
        using Mat canvasRegion = new Mat(canvas, canvasRoi);
        srcRegion.CopyTo(canvasRegion);

        Debug.Log($"[CenterMask] src={srcW}×{srcH} → canvas={canvasW}×{canvasH}  " +
                  $"paste@({dstX},{dstY}) size={copyW}×{copyH}");
        return canvas;
    }

    // ── 리소스 로드 헬퍼 ─────────────────────────────────────
    // prefix+"0", prefix+"1"... 순서로 로드, 최초 null에서 중단
    static Texture2D[] LoadTextureSequence(string prefix, int maxCount)
    {
        var list = new List<Texture2D>();
        for (int i = 0; i <= maxCount; i++)
        {
            var tex = Resources.Load<Texture2D>($"{prefix}{i}");
            if (tex == null) break;
            list.Add(tex);
        }
        return list.ToArray();
    }

    // 텍스처 자체는 필요없고 수량만 셀 때 사용 (로드 후 즉시 언로드)
    static int CountResources(string prefix, int maxCount)
    {
        int count = 0;
        while (count <= maxCount)
        {
            var tex = Resources.Load<Texture2D>($"{prefix}{count}");
            if (tex == null) break;
            Resources.UnloadAsset(tex);
            count++;
        }
        return count;
    }

    // ── 디버그 오버레이 ───────────────────────────────────────
    // • clone 제거: DrawDetectedMarkers를 bgr에 직접 그림
    // • BGR→RGB 1회 변환 후 Unity Flip, Texture2D/_overlayBuffer 재사용으로 GC 방지
    void DrawOverlay(Mat frame, Point2f[][] corners, int[] ids)
    {
        if (overlayView == null) return;

        using Mat bgr = new Mat();
        Cv2.CvtColor(frame, bgr, ColorConversionCodes.RGBA2BGR);
        CvAruco.DrawDetectedMarkers(bgr, corners, ids);

        // BGR→RGB + FlipMode.X (Unity 텍스처 좌표계 보정)
        using Mat rgb = new Mat();
        Cv2.CvtColor(bgr, rgb, ColorConversionCodes.BGR2RGB);
        Cv2.Flip(rgb, rgb, FlipMode.X);

        int W = rgb.Cols, H = rgb.Rows, byteLen = W * H * 3;
        if (_overlayBuffer == null || _overlayBuffer.Length != byteLen)
            _overlayBuffer = new byte[byteLen];
        Marshal.Copy(rgb.Data, _overlayBuffer, 0, byteLen);

        if (_overlayTex == null || _overlayTex.width != W || _overlayTex.height != H)
        {
            if (_overlayTex != null) Destroy(_overlayTex);
            _overlayTex = new Texture2D(W, H, TextureFormat.RGB24, false);
        }
        _overlayTex.LoadRawTextureData(_overlayBuffer);
        _overlayTex.Apply();
        overlayView.texture = _overlayTex;
    }

    // 저장 경로: Assets/Resources/FishTest/{timestamp}_fish{n}_{stepName}.png
    void SaveDebugMat(Mat mat, string stepName, int fishIndex)
    {
        if (!saveDebugImages || mat == null || mat.Empty()) return;

        string dir = System.IO.Path.Combine(Application.dataPath, "Resources", "FishTest");
        if (!System.IO.Directory.Exists(dir))
            System.IO.Directory.CreateDirectory(dir);

        string ts   = System.DateTime.Now.ToString("HHmmss_fff");
        string path = System.IO.Path.Combine(dir, $"{ts}_fish{fishIndex}_{stepName}.png");

        int W = mat.Cols, H = mat.Rows;

        // Mat → byte[] → Texture2D → EncodeToPNG (OpenCV ImWrite 코덱 불필요)
        Texture2D tex;
        if (mat.Channels() == 1)
        {
            // 1ch 마스크: Gray → RGB24 변환 후 저장
            using Mat rgb = new Mat();
            Cv2.CvtColor(mat, rgb, ColorConversionCodes.GRAY2RGB);
            Cv2.Flip(rgb, rgb, FlipMode.X);
            byte[] buf = new byte[W * H * 3];
            Marshal.Copy(rgb.Data, buf, 0, buf.Length);
            tex = new Texture2D(W, H, TextureFormat.RGB24, false);
            tex.LoadRawTextureData(buf);
        }
        else
        {
            // 4ch RGBA: 그대로 RGBA32 저장
            using Mat flipped = new Mat();
            Cv2.Flip(mat, flipped, FlipMode.X);
            byte[] buf = new byte[W * H * 4];
            Marshal.Copy(flipped.Data, buf, 0, buf.Length);
            tex = new Texture2D(W, H, TextureFormat.RGBA32, false);
            tex.LoadRawTextureData(buf);
        }

        tex.Apply();
        System.IO.File.WriteAllBytes(path, tex.EncodeToPNG());
        DestroyImmediate(tex);

        Debug.Log($"[DebugSave] {stepName} → {path}");
    }
}

// ── 방향별 마스크 오프셋 세트 ─────────────────────────────────────────────
// Unity Inspector 에서 fish 별로 4 방향 오프셋을 각각 설정합니다.
// anchorPos: 0=정면(0°), 1=좌90°(CCW), 2=180°, 3=우90°(CW)
[System.Serializable]
public class MaskOffsetSet
{
    [Tooltip("정면 (0°) 오프셋. x=좌(-)/우(+), y=위(-)/아래(+) (픽셀)")]
    public Vector2Int front      = Vector2Int.zero;

    [Tooltip("좌측 90° (CCW) 오프셋")]
    public Vector2Int left90     = Vector2Int.zero;

    [Tooltip("180° 오프셋")]
    public Vector2Int rotate180  = Vector2Int.zero;

    [Tooltip("우측 90° (CW) 오프셋")]
    public Vector2Int right90    = Vector2Int.zero;

    public Vector2Int GetOffset(int anchorPos)
    {
        return anchorPos switch
        {
            1 => left90,
            2 => rotate180,
            3 => right90,
            _ => front,   // 0 또는 범위 밖
        };
    }
}

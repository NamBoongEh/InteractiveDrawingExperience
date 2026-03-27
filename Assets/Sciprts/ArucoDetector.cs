using UnityEngine;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
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

    [Header("Output Size (per fish)")]
    [Tooltip("fish별 고정 출력 크기. 인덱스=fishIndex.\n" +
             "x=가로, y=세로. 0으로 두면 자동 계산(자연크기×2).\n" +
             "플레이 중 'Auto-Calibrate' 컨텍스트 메뉴로 중간값 자동 측정 가능.")]
    public Vector2Int[] fishOutputSizes = new Vector2Int[]
    {
        new Vector2Int(2200, 1330),   // fish0
    };

    [Header("Physical Dimensions (mm) — 비율 기반 마스크 크기")]

#if UNITY_SENTIS
    [Header("AI Segmentation (Unity Sentis)")]
    [Tooltip("U2Net 살리언시 모델로 물고기 영역 추출. aiSegModel에 ONNX 에셋 할당 필요.")]
    public bool useAISegmentation = false;
    [Tooltip("Unity Sentis에서 실행할 ONNX 모델 에셋 (u2netp.onnx 권장).")]
    public ModelAsset aiSegModel;
    [Tooltip("추론 백엔드. GPUCompute = GPU(빠름), CPU = 호환성 높음.")]
    public BackendType aiBackend = BackendType.GPUCompute;
    [Range(128, 512)]
    [Tooltip("모델 입력 해상도. U2Net-p = 320 고정.")]
    public int aiInputSize = 320;
    [Range(0f, 1f)]
    [Tooltip("살리언시 이진화 임계값. 높을수록 마스크 좁아짐 (0.4~0.6 권장).")]
    public float aiThreshold = 0.0341f;
    [Range(0f, 1f)]
    [Tooltip("중앙 가우시안 σ. 외부 그림 억제 강도. 0=비활성화, 0.35=권장.")]
    public float aiCenterSigma = 0.041f;
    [Range(0, 40)]
    [Tooltip("마스크 침식(px). 마스크가 그림보다 크게 나올 때 줄임 (8~15 권장).")]
    public int aiMaskErodePx = 1;
    [Tooltip("꼬리 내부 구멍(hollow) 자동 채우기. 꼬리 안쪽이 투명해지는 문제 해결.")]
    public bool aiHoleFill = true;

    [Header("Contour Segmentation (흰 배경 기반)")]
    [Tooltip("AI 대신 흰 배경 컨투어로 물고기 영역 추출. 모델 불필요.")]
    public bool useContourSegmentation = false;
    [Range(21, 301)]
    [Tooltip("AdaptiveThreshold 블록 크기(홀수). 로컬 영역 크기.\n" +
             "클수록 넓은 조명 불균일에 강함. 2200px 이미지 기준 101~201 권장.")]
    public int contourAdaptBlock = 151;
    [Range(1, 60)]
    [Tooltip("AdaptiveThreshold C값. 로컬 평균에서 이 값만큼 더 어두운 픽셀만 선으로 감지.\n" +
             "높이면 선이 줄고, 낮추면 선이 늘어남 (15~25 권장).")]
    public int contourAdaptC = 35;
    [Range(1, 40)]
    [Tooltip("외곽선 팽창(px). 끊어진 선을 이어 닫힌 윤곽 만들기.")]
    public int contourDilatePx = 2;
    [Range(0, 30)]
    [Tooltip("마스크 침식(px). 팽창으로 두꺼워진 외곽선 두께 보정.")]
    public int contourErodePx  = 1;
    [Range(10, 100)]
    [Tooltip("클로징 커널(px). 물고기 내부 흰 구멍 채우기.")]
    public int contourClosePx  = 10;
    [Range(0, 50)]
    [Tooltip("[방법 4] FindContours 전 Gaussian 스무딩(px).\n" +
             "JPEG 노이즈·들쭉날쭉 외곽선 완화. 0=비활성. 15~25 권장.")]
    public int contourPreSmoothPx = 0;
#else
    [Header("AI Segmentation  ※ com.unity.sentis 패키지 미설치")]
    [Tooltip("Package Manager → Add by name → com.unity.sentis 설치 후 사용 가능.")]
    public bool useAISegmentation = false;
#endif

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
    private const string kOnlyFishPrefix = "FishMaskOnlyFish/fish";  // 생물 실루엣 전용
    private const string kFishPrefix     = "Fish/fish";
    private const int    kMaxFishCount   = 20;

    // ── FishMaskApplier/ 폴더: AI 세그멘테이션 후 AND 마스크 적용
    // fishApplierSources.Length = fish 수 (fishCount 필드 불필요)
    private Texture2D[] fishApplierSources;
    private Mat[]       _fishApplierMasks;   // Start()에서 1회 변환·캐시

    // ── FishMaskOnlyFish/ 폴더: 생물 실루엣 전용 마스크 (배경 없이 딱 생물만)
    // inkBBox 기반 스케일링으로 그림 위치에 독립적으로 적용
    private Mat[]       _fishOnlyMasks;
    private float[]     _fishOnlyAspects; // 템플릿 tight-crop W/H 비율 (출력 패딩용)

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
    private byte[]    _distByteBuf;       // DistanceTransform float 바이트 버퍼
    private byte[]    _alphaRowBuf;       // DistanceTransform 알파 결과 버퍼
#if UNITY_SENTIS
    private byte[]    _aisegRawBuf;       // AISeg 전처리 원시 바이트 버퍼
    private float[]   _aisegFloatBuf;     // AISeg 전처리 float 버퍼
    private byte[]    _aisegPixelBuf;     // AISeg_SaliencyToMat 결과 픽셀 버퍼
#endif

    private static readonly string[] s_rotLabels = { "회전 없음", "90°CCW", "180°", "90°CW" };

    // ── 초기화 ────────────────────────────────────────────────
    void Start()
    {
        cam = ScannerManager.Instance.webCam;
        arucoDict = CvAruco.GetPredefinedDictionary(PredefinedDictionaryName.Dict4X4_50);
        detParams = DetectorParameters.Create();
        SetFailureMessage(null);

        // FishMaskApplier/ → fish 수(종류) 결정 + 컨투어 실패 시 폴백 마스크용
        // ※ 컨투어 세그멘테이션 정상 경로에서는 사용하지 않음.
        //   (고정 템플릿 AND는 그림 위치·크기 편차로 인한 클리핑 부작용 발생 → 삭제됨)
        //   컨투어 검출 자체가 실패한 경우에만 ApplyFishTemplateMask()로 폴백.
        fishApplierSources = LoadTextureSequence(kApplierPrefix, kMaxFishCount);
        if (fishApplierSources.Length == 0)
        {
            // Fish/ 폴더로 그룹 수 파악 — 마스크 없음 (null 배열)
            int count = CountResources(kFishPrefix, kMaxFishCount);
            fishApplierSources = new Texture2D[count]; // 전부 null, 크기만 맞춤
        }
        else
        {
            for (int i = 0; i < fishApplierSources.Length; i++)
                Debug.Log($"[ArucoDetector] FishMaskApplier fish{i}: {fishApplierSources[i].width}×{fishApplierSources[i].height}");
        }

        // FishMaskApplier 텍스처 → 1채널 마스크 Mat 캐시 (폴백 전용)
        _fishCount        = fishApplierSources.Length;
        _fishApplierMasks = new Mat[_fishCount];
        for (int i = 0; i < _fishCount; i++)
            if (fishApplierSources[i] != null)
                _fishApplierMasks[i] = ApplierTextureToMask(fishApplierSources[i]);

        // 마스크 변환 완료 → 원본 Texture2D 즉시 해제 (Mat 캐시만 보유)
        foreach (var tex in fishApplierSources)
            if (tex != null) Resources.UnloadAsset(tex);
        fishApplierSources = null;

        // FishMaskOnlyFish/ 로드: 생물 실루엣 전용 템플릿 (배경 없이 딱 생물 형태만)
        // inkBBox(실제 잉크 위치) 기반으로 스케일링 → 그림 위치에 독립적
        {
            Texture2D[] onlySrcs = LoadTextureSequence(kOnlyFishPrefix, kMaxFishCount);
            _fishOnlyMasks   = new Mat[_fishCount];
            _fishOnlyAspects = new float[_fishCount];
            for (int i = 0; i < Mathf.Min(onlySrcs.Length, _fishCount); i++)
                if (onlySrcs[i] != null)
                {
                    _fishOnlyMasks[i] = ApplierTextureToMask(onlySrcs[i]);
                    // tight-crop 비율 계산: 검은 여백 제외한 실루엣 W/H
                    using (Mat tPts = new Mat())
                    {
                        Cv2.FindNonZero(_fishOnlyMasks[i], tPts);
                        if (tPts != null && !tPts.Empty())
                        {
                            var tr = Cv2.BoundingRect(tPts);
                            _fishOnlyAspects[i] = (float)tr.Width / tr.Height;
                        }
                        else
                            _fishOnlyAspects[i] = 1f;
                    }
                    Debug.Log($"[ArucoDetector] FishMaskOnlyFish fish{i}: {onlySrcs[i].width}×{onlySrcs[i].height}  aspect={_fishOnlyAspects[i]:F3}");
                }
            foreach (var t in onlySrcs) if (t != null) Resources.UnloadAsset(t);
        }

        if (_fishCount == 0)
        {
            Debug.LogWarning("[ArucoDetector] Fish 리소스가 없습니다. FishMaskApplier/ 또는 Fish/ 폴더를 확인하세요.");
            return;
        }
        Debug.Log($"[ArucoDetector] Fish {_fishCount}개 로드 → ID 범위 0~{_fishCount * 4 - 1}");

        // ── Sentis 초기화 (U2Net) ────────────────────────────────
#if UNITY_SENTIS
        if (useAISegmentation && !useContourSegmentation)
        {
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
        if (_fishOnlyMasks != null)
            foreach (var m in _fishOnlyMasks) m?.Dispose();
        // fishApplierSources: Start()에서 마스크 변환 직후 이미 해제됨
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

            if (holdTimer >= holdDurationToCapture)
            {
                holdTimer = 0f;
                try
                {
                    var (warped, tlMarkerId) = Warp(frame, corners, ids);
                    SetFailureMessage(null);
                    OnAllMarkersDetected?.Invoke(warped, tlMarkerId);
                }
                catch (Exception e)
                {
                    Debug.LogError($"[ArucoDetector] 이미지 생성 실패: {e.Message}");
                    SetFailureMessage("이미지 생성에 실패했습니다.\n다시 카메라에 인식해 주세요.");
                }
            }
        }
        else
        {
            holdTimer = 0f;
        }
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

    // ── 워프 → 회전 → 마스크 ─────────────────────────────────
    // 출력 크기를 고정값 없이 카메라 이미지 내 마커 꼭짓점 간 거리로 자연 계산
    (Texture2D tex, int tlMarkerId) Warp(Mat frame, Point2f[][] corners, int[] ids)
    {
        var (src, anchorPos, tlMarkerId) = GetWarpParams(corners, ids);
        int fishIndex = tlMarkerId / 4;

        // ① 자연 크기: 카메라 픽셀 기준 종이 가로·세로 길이
        var (warpW, warpH) = ComputeNaturalSize(src);
        Debug.Log($"[Warp] 자연 크기: {warpW}×{warpH}  ratio={warpW/(float)warpH:F3}");

        // ② 출력 크기: fishOutputSizes[fishIndex] 우선, 없으면 자연크기×2
        int outW, outH;
        if (fishOutputSizes != null && fishIndex < fishOutputSizes.Length
            && fishOutputSizes[fishIndex].x > 0 && fishOutputSizes[fishIndex].y > 0)
        {
            outW = fishOutputSizes[fishIndex].x;
            outH = fishOutputSizes[fishIndex].y;
        }
        else
        {
            outW = warpW * 2;
            outH = warpH * 2;
        }
        Debug.Log($"[Warp] 출력 크기: {outW}×{outH}  (fish{fishIndex})");

        Point2f[] dst = {
            new Point2f(0,    0),
            new Point2f(outW, 0),
            new Point2f(outW, outH),
            new Point2f(0,    outH)
        };
        using Mat M = Cv2.GetPerspectiveTransform(src, dst);
        using Mat warped = new Mat();
        // BorderTypes.Constant + 흰색: 종이 바깥 영역이 검정이 아닌 흰색으로 채워짐
        Cv2.WarpPerspective(frame, warped, M, new OpenCvSharp.Size(outW, outH),
                            InterpolationFlags.Cubic, BorderTypes.Constant,
                            new Scalar(255, 255, 255, 255));
        Debug.Log($"[Warp] ① warped : {warped.Cols}×{warped.Rows}  (자연={warpW}×{warpH} → 고정={outW}×{outH})");

        // 노이즈 제거 생략: 세그멘테이션 내부 GaussianBlur가 충분히 처리
        // (BilateralFilter d=3 → ~80ms 절약, 시각 품질 영향 미미)
        SaveDebugMat(warped, "00_warp_raw", fishIndex);

        if (anchorPos == 0)
        {
            ApplyMask(warped, fishIndex);
            SaveDebugMat(warped, "07_after_mask", fishIndex);
            Debug.Log($"[Warp] 최종 출력: {warped.Cols}×{warped.Rows} ({s_rotLabels[0]})");
            return (FinalizeOutput(warped, fishIndex), tlMarkerId);
        }

        RotateFlags flag = anchorPos == 1 ? RotateFlags.Rotate90CounterClockwise
                         : anchorPos == 2 ? RotateFlags.Rotate180
                         :                  RotateFlags.Rotate90Clockwise;
        using Mat rotated = new Mat();
        Cv2.Rotate(warped, rotated, flag);

        ApplyMask(rotated, fishIndex);
        SaveDebugMat(rotated, "07_after_mask", fishIndex);
        Debug.Log($"[Warp] 최종 출력: {rotated.Cols}×{rotated.Rows} ({s_rotLabels[anchorPos]})");
        return (FinalizeOutput(rotated, fishIndex), tlMarkerId);
    }

    // ── Texture2D 변환 ─────────────────────────────────────────
    // ApplyMask(Otsu + 컨투어 내부 채우기) alpha 기준 tight crop → 2× 업스케일 → 반환
    Texture2D FinalizeOutput(Mat mat, int fishIndex)
    {
        SaveDebugMat(mat, "08_final", fishIndex);

        // alpha 기준 tight crop
        using Mat a = new Mat();
        Cv2.ExtractChannel(mat, a, 3);
        using Mat nz = new Mat();
        Cv2.FindNonZero(a, nz);
        if (nz != null && !nz.Empty())
        {
            var bbox = Cv2.BoundingRect(nz);
            int pad = Mathf.Max(5, Mathf.Max(mat.Cols, mat.Rows) / 200);
            int x = Mathf.Max(0, bbox.X - pad);
            int y = Mathf.Max(0, bbox.Y - pad);
            int w = Mathf.Min(mat.Cols - x, bbox.Width  + pad * 2);
            int h = Mathf.Min(mat.Rows - y, bbox.Height + pad * 2);
            if (w > 0 && h > 0)
            {
                using Mat cropped = new Mat(mat, new OpenCvSharp.Rect(x, y, w, h));
                using Mat upscaled = new Mat();
                Cv2.Resize(cropped, upscaled,
                           new OpenCvSharp.Size(w * 2, h * 2),
                           0, 0, InterpolationFlags.Cubic);
                SaveDebugMat(upscaled, "09_cropped", fishIndex);
                Debug.Log($"[FinalizeOutput] {mat.Cols}×{mat.Rows} → {w}×{h} → {w*2}×{h*2}  fish{fishIndex}");
                return MatToTexture2D(upscaled);
            }
        }
        return MatToTexture2D(mat);
    }

    // ── 마커 내부 꼭짓점 거리로 자연 워프 크기 계산 ──────────
    // src = [TL, TR, BR, BL] (GetWarpParams 반환 순서)
    static (int w, int h) ComputeNaturalSize(Point2f[] src)
    {
        static float Dist(Point2f a, Point2f b)
        {
            float dx = b.X - a.X, dy = b.Y - a.Y;
            return Mathf.Sqrt(dx * dx + dy * dy);
        }
        float topW    = Dist(src[0], src[1]); // TL → TR
        float bottomW = Dist(src[3], src[2]); // BL → BR
        float leftH   = Dist(src[0], src[3]); // TL → BL
        float rightH  = Dist(src[1], src[2]); // TR → BR
        int w = Mathf.Max(1, Mathf.RoundToInt((topW + bottomW) * 0.5f));
        int h = Mathf.Max(1, Mathf.RoundToInt((leftH + rightH) * 0.5f));
        return (w, h);
    }

    void ApplyMask(Mat rgba, int fishIndex)
    {
        // ── HSV V채널 기반 검정 외곽선 전용 검출 → 컨투어 채우기 ───────────
        // Otsu(gray) 대신 V채널을 사용하는 이유:
        //   · 검정 잉크:      V ≈ 0~40   (매우 어두움)
        //   · 파란 크레파스:  V ≈ 80~200 (max(R,G,B) = B값이 높음)
        //   · 갈색 연필:      V ≈ 100~180
        //   → V < 70 임계값으로 검정 외곽선만 분리, 컬러 블롭이 컨투어에서 제외
        // ─────────────────────────────────────────────────────────────────────
        using Mat bgr = new Mat();
        Cv2.CvtColor(rgba, bgr, ColorConversionCodes.RGBA2BGR);
        using Mat hsv = new Mat();
        Cv2.CvtColor(bgr, hsv, ColorConversionCodes.BGR2HSV);

        // V 채널 추출 후 임계값: V < 70 = 검정 잉크, V ≥ 70 = 컬러/흰 배경
        using Mat vCh = new Mat();
        Cv2.ExtractChannel(hsv, vCh, 2);
        using Mat blackMask = new Mat();
        Cv2.Threshold(vCh, blackMask, 70, 255, ThresholdTypes.BinaryInv);

        // Dilate: 끊어진 외곽선 연결
        int kSz = Mathf.Max(3, rgba.Cols / 300) | 1;
        using (var dk = Cv2.GetStructuringElement(
                   MorphShapes.Ellipse, new OpenCvSharp.Size(kSz, kSz)))
            Cv2.Dilate(blackMask, blackMask, dk, iterations: 2);

        // 가장 큰 컨투어(생물 외곽선) 내부 채우기 → alpha
        using Mat alpha = new Mat(rgba.Rows, rgba.Cols, MatType.CV_8UC1, Scalar.Black);
        Cv2.FindContours(blackMask, out Point[][] ctrs, out _,
                         RetrievalModes.External,
                         ContourApproximationModes.ApproxSimple);
        if (ctrs.Length > 0)
        {
            int maxI = 0; double maxA = 0;
            for (int i = 0; i < ctrs.Length; i++)
            {
                double a = Cv2.ContourArea(ctrs[i]);
                if (a > maxA) { maxA = a; maxI = i; }
            }
            Cv2.DrawContours(alpha, ctrs, maxI, Scalar.White, thickness: -1);
        }

        Cv2.MixChannels(new[] { alpha }, new[] { rgba }, s_alphaMixMap);
    }

    // ════════════════════════════════════════════════════════════
    // ── [Contour Segmentation] 검정 외곽선 장벽 + 코너 FloodFill
    // ════════════════════════════════════════════════════════════
    // 핵심 원리: 굵은 검정 외곽선만 장벽으로 추출 → 코너 FloodFill →
    //   외곽선 밖 볼펜 흔적은 배경으로 마킹 → 마스크에서 제외.
    //
    // 블러 근거 (W=2200, blurSz≈11px):
    //   볼펜 1-2px 스트로크 → 블러 후 피크 ≈ 214 → threshold(80) 통과 ❌
    //   검정 외곽선 10+px   → 블러 후 피크 ≈  50 → threshold(80) 통과 ✅
    // ════════════════════════════════════════════════════════════
    // ── [Contour Segmentation] 전면 재설계 — FloodFill 의존 제거
    //    그림 픽셀 직접 감지 → Large Close → 실루엣 추출
    // ════════════════════════════════════════════════════════════
    void ApplyContourSegmentation(Mat rgba, int fishIndex)
    {
        int W = rgba.Cols, H = rgba.Rows;

        // ══ 핵심 최적화: 세그멘테이션은 70% 크기(1/2 픽셀)에서 수행 ══
        // W/2(25% 픽셀) → W*0.7(49% 픽셀): 픽셀 수 약 2배, 모폴로지 품질 향상
        // 최종 alphaMask만 원본 크기로 업스케일하여 적용
        int sW = W * 7 / 10, sH = H * 7 / 10;
        using Mat small = new Mat();
        Cv2.Resize(rgba, small, new OpenCvSharp.Size(sW, sH),
                   0, 0, InterpolationFlags.Linear);
        Debug.Log($"[Contour] 입력:{W}×{H} → 처리:{sW}×{sH}  fish{fishIndex}");

        // ① 그레이스케일 (small 기준)
        using Mat sbgr = new Mat();
        Cv2.CvtColor(small, sbgr, ColorConversionCodes.RGBA2BGR);
        using Mat sgray = new Mat();
        Cv2.CvtColor(sbgr, sgray, ColorConversionCodes.BGR2GRAY);

        // ② 임계값: Otsu 자동 계산 후 200 캡 적용
        // Otsu: 종이 위치·조명 무관하게 최적 분리 (배경지 추정 불필요)
        // 캡 200: Otsu가 너무 낮게 설정되어 밝은 내부 픽셀(연필·흰 몸통)을
        //         배경으로 오분류하는 현상 방지 → FillPoly 내부 잘림 해소
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

        // ②-b 동적 inkBBox: 채색(유채색) 영역 제거 + centroid 대칭 bbox
        // ────────────────────────────────────────────────────────────────────
        // 근본 원인: 파란색·빨간색 크레파스 등 채색 영역이 drawMask에 포함되어
        //   inkBBox가 오른쪽/위로 치우침 → 템플릿 배치 오프셋 → 물고기 편향 출력
        // 해결책 ①: HSV 채도(S채널) > 50 = 유채색 픽셀 → inkBBox 계산에서 제외
        //   검정 윤곽선: S ≈ 0~20 (무채색) → 유지
        //   파란색 크레파스: S ≈ 100~200 → 제거
        // 해결책 ②: 컨투어 centroid 기반 대칭 bbox → 잔여 blob의 오프셋 보정
        var inkBBox = new OpenCvSharp.Rect(sW / 8, sH / 8, sW * 6 / 8, sH * 6 / 8);
        {
            // ── ① 유채색 픽셀 마스크 생성 ──────────────────────────────
            using Mat hsv       = new Mat();
            using Mat satCh     = new Mat();
            using Mat colorMask = new Mat();
            Cv2.CvtColor(sbgr, hsv, ColorConversionCodes.BGR2HSV);
            Cv2.ExtractChannel(hsv, satCh, 1);                       // S 채널
            // S>100: 뚜렷한 채색만 제거 (S<100인 먹색 윤곽선 보존)
            // 이전 S>50은 검정 윤곽선(먹색, S≈30~70)까지 제거해 bboxSource가 빈 이미지가 됨
            Cv2.Threshold(satCh, colorMask, 100, 255, ThresholdTypes.Binary);
            // 팽창 최소화(sW/400 ≈ 1~2px): 경계 안티앨리어싱만 커버
            // 이전 sW/50=14px 팽창은 파란 영역 옆 검정 윤곽선까지 침식함
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
                // 위치: 색상필터 잉크 centroid (파란 blob·빨간 장식 미포함 → 정확한 생물 중심)
                int cx = (int)(sumX / sumArea);
                int cy = (int)(sumY / sumArea);
                // 크기: bboxClean union bbox (색상필터된 검정 윤곽선 범위)
                // 검정 윤곽선이 지느러미·등껍질 테두리까지 포함 → 채색 부위도 outline 내부에 있음
                // fullR(전체 drawMask) 대신 bboxClean 사용: 거북이 빨간 하트 등 분산 장식이 bbox 왜곡하는 문제 방지
                int hw = (ux2 - ux1) / 2;
                int hh = (uy2 - uy1) / 2;
                int nx = Mathf.Max(0, cx - hw);
                int ny = Mathf.Max(0, cy - hh);
                inkBBox = new OpenCvSharp.Rect(nx, ny,
                    Mathf.Min(sW - nx, hw * 2), Mathf.Min(sH - ny, hh * 2));
                Debug.Log($"[Contour] inkBBox(centroid+bboxClean) c=({cx},{cy}) hw={hw} hh={hh} → ({inkBBox.X},{inkBBox.Y}) {inkBBox.Width}×{inkBBox.Height}  fish{fishIndex}");
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

        // ③ FishMaskOnlyFish 기반 알파 직접 적용 (우선 경로)
        // ── 이전 방식과 차이점 ────────────────────────────────────────────────
        // 이전(FishMaskApplier AND): 템플릿을 전체 종이 크기로 늘림 → 위치 불일치 → 잘림
        // 현재(FishMaskOnlyFish): inkBBox(실제 잉크 위치) 기준으로 스케일링
        //   → 아이가 어디에 그려도 잉크 범위에 정확히 맞춤
        //   → 별 모양·ConvexHull 문제 없이 올바른 생물 실루엣 적용
        // ─────────────────────────────────────────────────────────────────────
        if (_fishOnlyMasks != null &&
            fishIndex < _fishOnlyMasks.Length &&
            _fishOnlyMasks[fishIndex] != null)
        {
            // ══════════════════════════════════════════════════════════════════
            // TemplateMask: FishMaskOnlyFish 템플릿을 출력 크기의 99% 비율로
            //               중앙에 배치하여 알파 마스크로 직접 사용
            // ──────────────────────────────────────────────────────────────────
            // inkBBox·컨투어 계산 없이 템플릿 실루엣만으로 마스킹
            // → 스캔 조건·채색 방식과 무관하게 항상 안정적 결과
            // ══════════════════════════════════════════════════════════════════

            Mat tmpl = _fishOnlyMasks[fishIndex]; // CV_8UC1 grayscale

            // ① 목표 크기: 출력 해상도의 99% (가로·세로 각각)
            int targetW = (int)(W * 0.99);
            int targetH = (int)(H * 0.99);

            // ② 템플릿 비율 유지 Fit-Inside 스케일
            double scaleX = (double)targetW / tmpl.Width;
            double scaleY = (double)targetH / tmpl.Height;
            double fitScale = System.Math.Min(scaleX, scaleY);
            int rw = (int)(tmpl.Width  * fitScale);
            int rh = (int)(tmpl.Height * fitScale);

            using Mat resized = new Mat();
            Cv2.Resize(tmpl, resized, new OpenCvSharp.Size(rw, rh),
                       0, 0, InterpolationFlags.Linear);
            Cv2.Threshold(resized, resized, 127, 255, ThresholdTypes.Binary);

            // ③ W×H 검은 캔버스 중앙에 배치
            using Mat alphaFull = new Mat(H, W, MatType.CV_8UC1, Scalar.Black);
            int ox = (W - rw) / 2;
            int oy = (H - rh) / 2;
            using (Mat roi = new Mat(alphaFull, new OpenCvSharp.Rect(ox, oy, rw, rh)))
                resized.CopyTo(roi);

            Cv2.MixChannels(new[] { alphaFull }, new[] { rgba }, s_alphaMixMap);
            Debug.Log($"[TemplateMask] 99% center fit  rw={rw} rh={rh}  fish{fishIndex}");
            SaveDebugMat(rgba, "ct_10_final", fishIndex);
            return;
        }

        // ── 이하: FishMaskOnlyFish 없을 때 모폴로지 파이프라인 폴백 ─────────
        // ④ Large Close: sW/25 ≈ 44px × 4회 (원본 88px × 4와 동일 상대 효과)
        using Mat closedBlob = new Mat();
        using (var ck = CreateEllipseKernel(sW, 25))
            Cv2.MorphologyEx(drawMask, closedBlob, MorphTypes.Close, ck, iterations: 4);
        SaveDebugMat(closedBlob, "ct_02_closed", fishIndex);

        // ⑤ Open: 잡음 제거
        using Mat cleanBlob = new Mat();
        using (var ok = CreateEllipseKernel(sW, 300))
            Cv2.MorphologyEx(closedBlob, cleanBlob, MorphTypes.Open, ok, iterations: 1);
        SaveDebugMat(cleanBlob, "ct_03_clean", fishIndex);

        // ⑥ 유효 컨투어 전체 수집
        // ── 핵심 변경: 단일 최대 컨투어만 선택 → 임계 면적 이상 ALL 수집 ──────
        // 거북이: 몸통 + 4개 지느러미가 잉크 선으로 분리된 별개 컨투어가 될 수 있음.
        // 단일 최대(몸통)만 FillPoly하면 지느러미는 Concave Close로 "팽창 도달"해야 함.
        // → 지느러미까지의 거리가 크면 아무리 큰 커널도 오목부를 채울 수 없음(별 모양 고착).
        // → 해결: 모든 유효 컨투어를 FillPoly로 먼저 채운 뒤 작은 Close로 틈새만 연결.
        Cv2.FindContours(cleanBlob, out Point[][] contours, out _,
            RetrievalModes.External, ContourApproximationModes.ApproxSimple);
        if (contours.Length == 0)
        {
            Debug.LogWarning("[Contour] 컨투어 없음 → FishMask 폴백");
            ApplyFishTemplateMask(rgba, fishIndex);
            return;
        }
        double minArea = (double)sW * sH * 0.005; // 0.5% 미만 잡음 제거
        var validContours = new System.Collections.Generic.List<Point[]>();
        double totalArea = 0;
        foreach (var c in contours)
        {
            double a = Cv2.ContourArea(c);
            if (a >= minArea) { validContours.Add(c); totalArea += a; }
        }
        if (validContours.Count == 0)
        {
            Debug.LogWarning("[Contour] 최소 면적 컨투어 없음 → FishMask 폴백");
            ApplyFishTemplateMask(rgba, fishIndex);
            return;
        }
        Debug.Log($"[Contour] 유효 컨투어 {validContours.Count}개  totalArea={totalArea:F0}px²  fish{fishIndex}");

        // ⑦ FillPoly: 유효 컨투어 전부 채우기 (몸통 + 지느러미 등 모두)
        using Mat alphaMask = Mat.Zeros(sH, sW, MatType.CV_8UC1);
        Cv2.FillPoly(alphaMask, validContours.ToArray(), Scalar.White);
        SaveDebugMat(alphaMask, "ct_04_fillpoly", fishIndex);

        // ⑧ FillHoles (small 크기)
        ContourFillHoles(alphaMask, sW, sH);
        SaveDebugMat(alphaMask, "ct_05_holefilled", fishIndex);

        // ⑨ 경계 스무딩 Close: 컨투어 틈새 연결 및 경계 매끄럽게 처리
        using (var concaveK = CreateEllipseKernel(sW, 15))
            Cv2.MorphologyEx(alphaMask, alphaMask, MorphTypes.Close, concaveK, iterations: 3);
        SaveDebugMat(alphaMask, "ct_06_concave", fishIndex);

        // ⑩ FishMaskApplier AND → 삭제됨
        // 이유: 고정 템플릿을 전체 영역으로 리사이즈 후 AND 하면
        //       그림 위치·크기·비율이 다를 때 지느러미/발이 잘리는 부작용 발생.
        //       inkBBox clip(⑫-b)이 더 정확하게 동일한 역할을 수행하므로 불필요.
        //       ApplyFishTemplateMask는 컨투어 검출 실패 시 폴백으로만 유지.

        // ⑪ 경계 스무딩 (small 크기)
        int smSz = Mathf.Max(5, (sW / 60) | 1);
        Cv2.GaussianBlur(alphaMask, alphaMask, new OpenCvSharp.Size(smSz, smSz), 0);
        Cv2.Threshold(alphaMask, alphaMask, 127, 255, ThresholdTypes.Binary);

        // ⑫ alphaMask 원본 크기로 업스케일 → 원본 rgba에 적용
        using Mat bigAlpha = new Mat();
        Cv2.Resize(alphaMask, bigAlpha, new OpenCvSharp.Size(W, H),
                   0, 0, InterpolationFlags.Linear);
        Cv2.Threshold(bigAlpha, bigAlpha, 127, 255, ThresholdTypes.Binary);

        // ⑫-b ★ 잉크 경계 클립: 실제 잉크 범위(inkBBox) 바깥 알파 강제 제거
        // Concave Close 과팽창 / 종이 여백 포함 문제를 구조적으로 차단
        // inkPad: 잉크 바깥 허용 여유 (볼펜 선폭 + 약간의 여유 = W/60 ≈ 1.7%)
        {
            int inkPad = Mathf.Max(5, W / 60);
            int clipX = Mathf.Max(0,     inkBBox.X      * W / sW - inkPad);
            int clipY = Mathf.Max(0,     inkBBox.Y      * H / sH - inkPad);
            int clipW = Mathf.Min(W - clipX, inkBBox.Width  * W / sW + inkPad * 2);
            int clipH = Mathf.Min(H - clipY, inkBBox.Height * H / sH + inkPad * 2);

            using Mat clipMask = Mat.Zeros(H, W, MatType.CV_8UC1);
            using (Mat roi = new Mat(clipMask, new OpenCvSharp.Rect(clipX, clipY, clipW, clipH)))
                roi.SetTo(Scalar.White);

            Cv2.BitwiseAnd(bigAlpha, clipMask, bigAlpha);
            Debug.Log($"[Contour] inkClip(full)=({clipX},{clipY}) {clipW}×{clipH}  pad={inkPad}  fish{fishIndex}");
        }

        Cv2.MixChannels(new[] { bigAlpha }, new[] { rgba }, s_alphaMixMap);
        Debug.Log($"[Contour] 완료: fish{fishIndex}  totalArea={totalArea:F0}px²(small)  contours={validContours.Count}");
        SaveDebugMat(rgba, "ct_10_final", fishIndex);
    }

    // ── 안전 FillHoles: all-white 버그 방지 ─────────────────────
    // 내부 구멍이 없거나 면적이 50% 초과(잘못된 마스크)이면 건너뜀
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

        // FloodFill 후: 남은 흰 픽셀 = 진짜 내부 구멍 (외부와 단절된 영역)
        // ★ 버그 수정: BitwiseNot 제거
        //   이전: BitwiseNot 후 CountNonZero → 항상 >50% → 무조건 skip
        //   수정: filled 그대로 CountNonZero(구멍 면적만) → 정상 동작
        if (Cv2.CountNonZero(filled) < (long)W * H * 0.5)
            Cv2.BitwiseOr(binaryMask, filled, binaryMask);
        else
            Debug.LogWarning("[ContourFillHoles] 구멍 면적 과다 → FillHoles 건너뜀 (all-white 방지)");
    }

    // ── FishMaskApplier 템플릿 마스크를 알파 채널에 직접 적용 ────
    // 캐시된 _fishApplierMasks 사용. 반환값: 적용 성공 여부
    bool ApplyFishTemplateMask(Mat rgba, int fishIndex)
    {
        if (_fishApplierMasks == null ||
            fishIndex >= _fishApplierMasks.Length ||
            _fishApplierMasks[fishIndex] == null)
            return false;

        int W = rgba.Cols, H = rgba.Rows;
        using Mat scaledMask = new Mat();
        Cv2.Resize(_fishApplierMasks[fishIndex], scaledMask, new OpenCvSharp.Size(W, H));
        Cv2.Threshold(scaledMask, scaledMask, 128, 255, ThresholdTypes.Binary);
        Cv2.MixChannels(new[] { scaledMask }, new[] { rgba }, s_alphaMixMap);
        Debug.Log($"[FishTemplate] 마스크 적용: {W}×{H}  Fish{fishIndex}");
        return true;
    }

    // ── 해상도 비례 타원형 구조 요소 생성 ────────────────────
    // radius = Max(1, dimension / divisor), 반환된 Mat은 호출자가 using 으로 관리
    static Mat CreateEllipseKernel(int dimension, int divisor)
    {
        int radius = Mathf.Max(1, dimension / divisor);
        return Cv2.GetStructuringElement(
            MorphShapes.Ellipse,
            new OpenCvSharp.Size(radius * 2 + 1, radius * 2 + 1));
    }

#if UNITY_SENTIS
    // ════════════════════════════════════════════════════════════
    // ── [AI Segmentation] Unity Sentis + 살리언시 모델
    // ════════════════════════════════════════════════════════════
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
        Cv2.Threshold(resized, alphaMask, (int)(aiThreshold * 255), 255, ThresholdTypes.Binary);
        Debug.Log($"[AISeg] ③ 이진화(threshold={aiThreshold:F2}): " +
                  $"마스크={Cv2.CountNonZero(alphaMask)}px / {W * H}px");
        SaveDebugMat(alphaMask, "03_binary_thresh", fishIndex);

        // ④ 마스크 침식: AI가 그림보다 크게 잡는 문제 보정
        if (aiMaskErodePx > 0)
        {
            int eks = aiMaskErodePx * 2 + 1;
            using Mat ek = Cv2.GetStructuringElement(MorphShapes.Ellipse,
                new OpenCvSharp.Size(eks, eks));
            Cv2.Erode(alphaMask, alphaMask, ek);
            Debug.Log($"[AISeg] ④ 침식({aiMaskErodePx}px) 후: {Cv2.CountNonZero(alphaMask)}px");
            SaveDebugMat(alphaMask, "04_after_erode", fishIndex);
        }

        // ⑤ 꼬리·지느러미 내부 구멍 채우기 (FloodFill 역방향)
        if (aiHoleFill)
        {
            AISeg_FillHoles(alphaMask);
            Debug.Log($"[AISeg] ⑤ 구멍채우기 후: {Cv2.CountNonZero(alphaMask)}px");
            SaveDebugMat(alphaMask, "05_after_holefill", fishIndex);
        }

        // ⑥ FishMaskApplier AND 제약: 외부 그림(붉은색 등) 차단 (항상 적용)
        //    마스크 파일 픽셀 크기를 "실제 종이 기준 해상도"로 사용
        //    공식: 실제 종이 px(마스크 파일) : 캡처 종이 px(W×H) = 동일 비율
        //    → scale = min(W/maskCols, H/maskRows), 비율 유지 + 중앙 배치
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
        }
        else
        {
            Debug.LogWarning($"[AISeg] ⑥ FishMaskApplier fish{fishIndex} 없음 → AND 건너뜀");
        }

        // ⑦ 모폴로지 클로징: 마스크 내부 작은 구멍·테이프 선 잔여물 제거
        {
            using Mat ck = Cv2.GetStructuringElement(MorphShapes.Ellipse,
                new OpenCvSharp.Size(15, 15));
            Cv2.MorphologyEx(alphaMask, alphaMask, MorphTypes.Close, ck);
        }

        // ⑧ 알파 경계 anti-alias: 5×5 GaussianBlur로 hard edge → 부드러운 전환
        Cv2.GaussianBlur(alphaMask, alphaMask, new OpenCvSharp.Size(5, 5), 1.5);

        Cv2.MixChannels(new[] { alphaMask }, new[] { rgba }, s_alphaMixMap);
        Debug.Log($"[AISeg] ⑨ 완료: {W}×{H}  Fish{fishIndex}");
    }

    // ── 내부 구멍 채우기 ─────────────────────────────────────
    // 이진 마스크에서 외부와 연결되지 않은 검정 영역(구멍)을 흰색으로 채움
    // 원리: 반전 → 코너 FloodFill(배경 마킹) → 재반전 → OR 병합
    static void AISeg_FillHoles(Mat binaryMask)
    {
        using Mat inv = new Mat();
        Cv2.BitwiseNot(binaryMask, inv);

        using Mat filled = inv.Clone();
        using Mat ffMask = Mat.Zeros(inv.Rows + 2, inv.Cols + 2, MatType.CV_8UC1);
        // 4코너에서 flood fill → 외부 배경을 검정(0)으로 마킹
        Cv2.FloodFill(filled, ffMask, new Point(0, 0),                        Scalar.Black);
        Cv2.FloodFill(filled, ffMask, new Point(inv.Cols - 1, 0),             Scalar.Black);
        Cv2.FloodFill(filled, ffMask, new Point(0, inv.Rows - 1),             Scalar.Black);
        Cv2.FloodFill(filled, ffMask, new Point(inv.Cols - 1, inv.Rows - 1),  Scalar.Black);

        // 남은 흰 픽셀 = 내부 구멍
        Cv2.BitwiseNot(filled, filled);
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
    // src(1채널)를 canvasW×canvasH 캔버스 중앙에 배치.
    //   src > 캔버스  → 중앙 크롭
    //   src < 캔버스  → 중앙 패드 (바깥은 0=투명)
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

    // ════════════════════════════════════════════════════════════
    // ── 디버그 이미지 저장
    // saveDebugImages=true 일 때만 동작.
    // 저장 경로: Assets/Resources/FishTest/{timestamp}_fish{n}_{stepName}.png
    // 1ch(마스크) → grayscale PNG / 4ch(RGBA) → BGRA PNG(투명도 포함)
    // ════════════════════════════════════════════════════════════
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

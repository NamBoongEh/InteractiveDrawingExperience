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
    public float aiThreshold = 0.5f;
    [Range(0f, 1f)]
    [Tooltip("중앙 가우시안 σ. 외부 그림 억제 강도. 0=비활성화, 0.35=권장.")]
    public float aiCenterSigma = 0.35f;
    [Range(0, 40)]
    [Tooltip("마스크 침식(px). 마스크가 그림보다 크게 나올 때 줄임 (8~15 권장).")]
    public int aiMaskErodePx = 10;
    [Tooltip("꼬리 내부 구멍(hollow) 자동 채우기. 꼬리 안쪽이 투명해지는 문제 해결.")]
    public bool aiHoleFill = true;
#else
    [Header("AI Segmentation  ※ com.unity.sentis 패키지 미설치")]
    [Tooltip("Package Manager → Add by name → com.unity.sentis 설치 후 사용 가능.")]
    public bool useAISegmentation = false;
#endif
    [Header("Debug UI")]
    public UnityEngine.UI.RawImage overlayView;
    public UnityEngine.UI.Text failureText;

    public event Action<Texture2D, int> OnAllMarkersDetected;

    private WebCamCapture cam;
    private ArucoDict arucoDict;
    private DetectorParameters detParams;
    private float holdTimer = 0f;

    // ── Fish 공통 ─────────────────────────────────────────────
    private int fishCount = 0;

    // ── FishMaskApplier/ 폴더: AI 세그멘테이션 후 AND 마스크 적용
    private Texture2D[] fishApplierSources;

    // ── Unity Sentis ───────────────────────────────────────────
#if UNITY_SENTIS
    private Worker _sentisWorker;
#endif

    // ── 재사용 버퍼 ───────────────────────────────────────────
    private byte[] _rawBuffer;
    private readonly Dictionary<int, int> _idToIdx = new Dictionary<int, int>();

    private static readonly string[] s_rotLabels = { "회전 없음", "90°CCW", "180°", "90°CW" };

    // ── 초기화 ────────────────────────────────────────────────
    void Start()
    {
        cam = ScannerManager.Instance.webCam;
        arucoDict = CvAruco.GetPredefinedDictionary(PredefinedDictionaryName.Dict4X4_50);
        detParams = DetectorParameters.Create();
        SetFailureMessage(null);

        // fishCount: ArUco ID 그룹 수 결정 (FishMaskApplier/ 기준)
        var applierList = new List<Texture2D>();
        for (int i = 0; ; i++)
        {
            var tex = Resources.Load<Texture2D>($"FishMaskApplier/fish{i}");
            if (tex == null && i > 0) break; // 첫 번째가 없어도 계속 시도
            applierList.Add(tex);
            if (tex != null)
                Debug.Log($"[ArucoDetector] FishMaskApplier fish{i}: {tex.width}×{tex.height}");
            if (i >= 20) break; // 안전 상한
        }

        // FishMaskApplier 없으면 Fish/ 폴더로 수량 확인
        if (applierList.Count == 0 || applierList.TrueForAll(t => t == null))
        {
            for (int i = 0; ; i++)
            {
                var tex = Resources.Load<Texture2D>($"Fish/fish{i}");
                if (tex == null) break;
                applierList.Add(null); // 자리만 확보
                fishCount++;
            }
        }
        else
        {
            fishCount = applierList.Count;
        }

        fishApplierSources = applierList.ToArray();

        if (fishCount == 0)
        {
            Debug.LogWarning("[ArucoDetector] Fish 리소스가 없습니다. FishMaskApplier/ 또는 Fish/ 폴더를 확인하세요.");
            return;
        }
        Debug.Log($"[ArucoDetector] Fish {fishCount}개 로드 → ID 범위 0~{fishCount * 4 - 1}");

        // ── Sentis 초기화 ────────────────────────────────────────
#if UNITY_SENTIS
        if (useAISegmentation)
        {
            if (aiSegModel == null)
            {
                Debug.LogError("[AISeg] aiSegModel이 Inspector에 할당되지 않았습니다. " +
                               "ONNX 모델을 할당하거나 useAISegmentation을 끄세요.");
            }
            else
            {
                try
                {
                    var model     = ModelLoader.Load(aiSegModel);
                    _sentisWorker = new Worker(model, aiBackend);
                    Debug.Log($"[AISeg] Sentis Worker 초기화 완료 (backend={aiBackend}  inputSize={aiInputSize})");
                }
                catch (Exception e)
                {
                    Debug.LogError($"[AISeg] Worker 초기화 실패: {e.Message}");
                }
            }
        }
#endif
    }

    void OnDestroy()
    {
        if (overlayView != null && overlayView.texture != null)
            Destroy(overlayView.texture);
#if UNITY_SENTIS
        _sentisWorker?.Dispose();
#endif
    }

    // ── 매 프레임 감지 ────────────────────────────────────────
    void Update()
    {
        if (ScannerManager.Instance.State != ScannerState.Scanning) return;
        if (!cam.CamTexture.isPlaying) return;

        using Mat frame = WebCamToMat(cam.CamTexture);
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
                    int tlMarkerId = FindTLMarkerId(corners, ids);
                    Texture2D warped = Warp(frame, corners, ids, tlMarkerId);
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

    // ── WebCamTexture → RGBA Mat ──────────────────────────────
    Mat WebCamToMat(WebCamTexture webcam)
    {
        Color32[] pixels = webcam.GetPixels32();
        int needed = pixels.Length * 4;
        if (_rawBuffer == null || _rawBuffer.Length != needed)
            _rawBuffer = new byte[needed];
        for (int i = 0; i < pixels.Length; i++)
        {
            _rawBuffer[i * 4    ] = pixels[i].r;
            _rawBuffer[i * 4 + 1] = pixels[i].g;
            _rawBuffer[i * 4 + 2] = pixels[i].b;
            _rawBuffer[i * 4 + 3] = 255;
        }
        Mat mat = new Mat(webcam.height, webcam.width, MatType.CV_8UC4);
        Marshal.Copy(_rawBuffer, 0, mat.Data, needed);
        Cv2.Flip(mat, mat, FlipMode.X);
        return mat;
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
        byte[] raw = new byte[rgba.Rows * rgba.Cols * 4];
        Marshal.Copy(rgba.Data, raw, 0, raw.Length);
        Texture2D tex = new Texture2D(rgba.Cols, rgba.Rows, TextureFormat.RGBA32, false);
        tex.LoadRawTextureData(raw);
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

        for (int g = 0; g < fishCount; g++)
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

    // ── 내부 꼭짓점 (중심에 가장 가까운 모서리) ──────────────
    static Point2f[] GetInnerCorners(Point2f[][] corners, int[] ids)
    {
        var (centers, midX, midY) = ComputeCenters(corners, ids.Length);
        Point2f tl=default, tr=default, br=default, bl=default;
        for (int i = 0; i < ids.Length; i++)
        {
            Point2f inner = ClosestCorner(corners[i], midX, midY);
            if      (centers[i].X<=midX && centers[i].Y<=midY) tl=inner;
            else if (centers[i].X> midX && centers[i].Y<=midY) tr=inner;
            else if (centers[i].X> midX && centers[i].Y> midY) br=inner;
            else                                                bl=inner;
        }
        return new[] { tl, tr, br, bl };
    }

    static Point2f ClosestCorner(Point2f[] c, float tx, float ty)
    {
        Point2f best = c[0]; float minD = float.MaxValue;
        foreach (var p in c) { float d=(p.X-tx)*(p.X-tx)+(p.Y-ty)*(p.Y-ty); if(d<minD){minD=d;best=p;} }
        return best;
    }

    // ── TL 마커 ID ────────────────────────────────────────────
    static int FindTLMarkerId(Point2f[][] corners, int[] ids)
    {
        var (centers, midX, midY) = ComputeCenters(corners, ids.Length);
        for (int i = 0; i < ids.Length; i++)
            if (centers[i].X <= midX && centers[i].Y <= midY) return ids[i];
        return ids[0];
    }

    // ── 회전 오프셋 ───────────────────────────────────────────
    static int FindAnchorOffset(Point2f[][] corners, int[] ids)
    {
        var (centers, midX, midY) = ComputeCenters(corners, ids.Length);
        for (int i = 0; i < ids.Length; i++)
        {
            if (ids[i] % 4 != 0) continue;
            bool left = centers[i].X <= midX, top = centers[i].Y <= midY;
            if ( left &&  top) return 0;
            if (!left &&  top) return 1;
            if (!left && !top) return 2;
            return 3;
        }
        return 0;
    }

    // ── 워프 → 회전 → 마스크 ─────────────────────────────────
    // 출력 크기를 고정값 없이 카메라 이미지 내 마커 꼭짓점 간 거리로 자연 계산
    Texture2D Warp(Mat frame, Point2f[][] corners, int[] ids, int tlMarkerId)
    {
        int fishIndex = tlMarkerId / 4;
        Point2f[] src = GetInnerCorners(corners, ids);

        // ① 자연 크기: 카메라 픽셀 기준 종이 가로·세로 길이
        var (warpW, warpH) = ComputeNaturalSize(src);
        Debug.Log($"[Warp] 자연 크기: {warpW}×{warpH}  ratio={warpW/(float)warpH:F3}");

        Point2f[] dst = {
            new Point2f(0,     0),
            new Point2f(warpW, 0),
            new Point2f(warpW, warpH),
            new Point2f(0,     warpH)
        };
        using Mat M = Cv2.GetPerspectiveTransform(src, dst);
        using Mat warped = new Mat();
        Cv2.WarpPerspective(frame, warped, M, new OpenCvSharp.Size(warpW, warpH));
        Debug.Log($"[Warp] ① warped : {warped.Cols}×{warped.Rows}");

        int anchorPos = FindAnchorOffset(corners, ids);

        if (anchorPos == 0)
        {
            ApplyMask(warped, fishIndex);
            Debug.Log($"[Warp] 최종 출력: {warped.Cols}×{warped.Rows} ({s_rotLabels[0]})");
            return MatToTexture2D(warped);
        }

        RotateFlags flag = anchorPos == 1 ? (RotateFlags)2
                         : anchorPos == 2 ? RotateFlags.Rotate180
                         :                  RotateFlags.Rotate90Clockwise;
        using Mat rotated = new Mat();
        Cv2.Rotate(warped, rotated, flag);

        // 리사이즈 없이 회전 결과를 그대로 사용
        // anchorPos 1/3: warpH×warpW (세로↔가로 교환), anchorPos 2: warpW×warpH 유지
        ApplyMask(rotated, fishIndex);
        Debug.Log($"[Warp] 최종 출력: {rotated.Cols}×{rotated.Rows} ({s_rotLabels[anchorPos]})");
        return MatToTexture2D(rotated);
    }

    // ── 마커 내부 꼭짓점 거리로 자연 워프 크기 계산 ──────────
    // src = [TL, TR, BR, BL] (GetInnerCorners 반환 순서)
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
#if UNITY_SENTIS
        if (useAISegmentation && _sentisWorker != null)
        {
            ApplyAISegmentation(rgba, fishIndex);
            return;
        }
#endif
        // Sentis 미사용 또는 Worker 미초기화 시 경고만 출력
        Debug.LogWarning($"[ArucoDetector] AI Segmentation 비활성 상태. " +
                         "useAISegmentation=true + aiSegModel 할당 확인.");
        {
        }
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
        using Mat resized = new Mat();
        Cv2.Resize(salMat, resized, new OpenCvSharp.Size(W, H));

        using Mat alphaMask = new Mat();
        Cv2.Threshold(resized, alphaMask, (int)(aiThreshold * 255), 255, ThresholdTypes.Binary);
        Debug.Log($"[AISeg] ③ 이진화(threshold={aiThreshold:F2}): " +
                  $"마스크={Cv2.CountNonZero(alphaMask)}px / {W * H}px");

        // ④ 마스크 침식: AI가 그림보다 크게 잡는 문제 보정
        if (aiMaskErodePx > 0)
        {
            int eks = aiMaskErodePx * 2 + 1;
            using Mat ek = Cv2.GetStructuringElement(MorphShapes.Ellipse,
                new OpenCvSharp.Size(eks, eks));
            Cv2.Erode(alphaMask, alphaMask, ek);
            Debug.Log($"[AISeg] ④ 침식({aiMaskErodePx}px) 후: {Cv2.CountNonZero(alphaMask)}px");
        }

        // ⑤ 꼬리·지느러미 내부 구멍 채우기 (FloodFill 역방향)
        if (aiHoleFill)
        {
            AISeg_FillHoles(alphaMask);
            Debug.Log($"[AISeg] ⑤ 구멍채우기 후: {Cv2.CountNonZero(alphaMask)}px");
        }

        // ⑥ FishMaskApplier AND 제약: 외부 그림(붉은색 등) 차단 (항상 적용)
        if (fishApplierSources != null &&
            fishIndex < fishApplierSources.Length &&
            fishApplierSources[fishIndex] != null)
        {
            using Mat tmpl = ApplierTextureToMask(fishApplierSources[fishIndex]);
            using Mat scaledTmpl = new Mat();
            Cv2.Resize(tmpl, scaledTmpl, new OpenCvSharp.Size(W, H));
            Cv2.Threshold(scaledTmpl, scaledTmpl, 128, 255, ThresholdTypes.Binary);
            Cv2.BitwiseAnd(alphaMask, scaledTmpl, alphaMask);
            Debug.Log($"[AISeg] ⑥ FishMaskApplier AND 후: {Cv2.CountNonZero(alphaMask)}px");
        }
        else
        {
            Debug.LogWarning($"[AISeg] ⑥ FishMaskApplier fish{fishIndex} 없음 → AND 건너뜀");
        }

        Cv2.MixChannels(new[] { alphaMask }, new[] { rgba }, new[] { 0, 3 });
        Debug.Log($"[AISeg] ⑦ 완료: {W}×{H}  Fish{fishIndex}");
    }

    // ── 내부 구멍 채우기 ─────────────────────────────────────
    // 이진 마스크에서 외부와 연결되지 않은 검정 영역(구멍)을 흰색으로 채움
    // 원리: 반전 → 코너 FloodFill(배경 마킹) → 재반전 → OR 병합
    static void AISeg_FillHoles(Mat binaryMask)
    {
        using Mat inv = new Mat();
        Cv2.BitwiseNot(binaryMask, inv);

        Mat filled = inv.Clone();
        using Mat ffMask = Mat.Zeros(inv.Rows + 2, inv.Cols + 2, MatType.CV_8UC1);
        // 4코너에서 flood fill → 외부 배경을 검정(0)으로 마킹
        Cv2.FloodFill(filled, ffMask, new Point(0, 0),             Scalar.Black);
        Cv2.FloodFill(filled, ffMask, new Point(inv.Cols - 1, 0),             Scalar.Black);
        Cv2.FloodFill(filled, ffMask, new Point(0, inv.Rows - 1),             Scalar.Black);
        Cv2.FloodFill(filled, ffMask, new Point(inv.Cols - 1, inv.Rows - 1),  Scalar.Black);

        // 남은 흰 픽셀 = 내부 구멍
        Cv2.BitwiseNot(filled, filled);
        Cv2.BitwiseOr(binaryMask, filled, binaryMask);
        filled.Dispose();
    }

    static float[] AISeg_Preprocess(Mat rgba, int sz)
    {
        using Mat rgb = new Mat();
        Cv2.CvtColor(rgba, rgb, ColorConversionCodes.RGBA2RGB);
        using Mat resized = new Mat();
        Cv2.Resize(rgb, resized, new OpenCvSharp.Size(sz, sz), interpolation: InterpolationFlags.Linear);

        float[] data = new float[3 * sz * sz];
        float[] mean = { 0.485f, 0.456f, 0.406f };
        float[] std  = { 0.229f, 0.224f, 0.225f };

        for (int y = 0; y < sz; y++)
        for (int x = 0; x < sz; x++)
        {
            Vec3b px  = resized.At<Vec3b>(y, x);
            int   idx = y * sz + x;
            data[0 * sz * sz + idx] = (px[0] / 255f - mean[0]) / std[0];
            data[1 * sz * sz + idx] = (px[1] / 255f - mean[1]) / std[1];
            data[2 * sz * sz + idx] = (px[2] / 255f - mean[2]) / std[2];
        }
        return data;
    }

    Mat AISeg_SaliencyToMat(Tensor<float> tensor, int W, int H)
    {
        float cx = W * 0.5f, cy = H * 0.5f;
        float sigX = aiCenterSigma * W, sigY = aiCenterSigma * H;
        bool  useGauss = aiCenterSigma > 0.01f;

        byte[] pixels = new byte[H * W];
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
            // 알파 없으면 grayscale 반전 (어두운 윤곽선 안쪽 = 마스크)
            for (int i = 0; i < pixels.Length; i++)
                data[i] = (byte)(255 - ((pixels[i].r + pixels[i].g + pixels[i].b) / 3));
        }

        Mat mat = new Mat(tex.height, tex.width, MatType.CV_8UC1);
        Marshal.Copy(data, 0, mat.Data, data.Length);
        Cv2.Flip(mat, mat, FlipMode.X);
        Cv2.Threshold(mat, mat, 128, 255, ThresholdTypes.Binary);
        return mat;
    }

    // ── 디버그 오버레이 ───────────────────────────────────────
    void DrawOverlay(Mat frame, Point2f[][] corners, int[] ids)
    {
        if (overlayView == null) return;
        using Mat bgr = new Mat();
        Cv2.CvtColor(frame, bgr, ColorConversionCodes.RGBA2BGR);
        using Mat overlay = bgr.Clone();
        CvAruco.DrawDetectedMarkers(overlay, corners, ids);
        using Mat overlayRGBA = new Mat();
        Cv2.CvtColor(overlay, overlayRGBA, ColorConversionCodes.BGR2RGBA);
        if (overlayView.texture != null) Destroy(overlayView.texture);
        overlayView.texture = MatToTexture2D(overlayRGBA);
    }
}

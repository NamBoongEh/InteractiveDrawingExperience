// FishSentisSegmentation.cs
// Unity Sentis + 살리언시 모델(U2Net 등) 단독 테스트 스크립트
// ※ com.unity.sentis 패키지 미설치 시 #if UNITY_SENTIS 내 코드는 비활성화됨
//
// 사용법:
//   1. 빈 GameObject에 이 컴포넌트 추가
//   2. Inspector에서 segModel에 ONNX 에셋 할당 (u2netp.onnx 권장)
//   3. inputTexture에 테스트할 스캔 이미지(Texture2D) 할당
//   4. 런타임에서 [Run Segmentation] 버튼 누르거나 autoRunOnStart 켜기
//   5. Console 로그 + outputView RawImage로 결과 확인

using UnityEngine;
#if UNITY_SENTIS
using Unity.Sentis;
#endif
using OpenCvSharp;
using System.Runtime.InteropServices;

#if UNITY_SENTIS
public class FishSentisSegmentation : MonoBehaviour
{
    [Header("Model")]
    [Tooltip("ONNX 살리언시 모델 에셋. U2Net-small(u2netp.onnx) 권장.")]
    public ModelAsset segModel;
    [Tooltip("추론 백엔드. GPUCompute = GPU(빠름), CPU = 호환성 높음.")]
    public BackendType backend = BackendType.GPUCompute;
    [Range(128, 512)]
    [Tooltip("모델 입력 해상도. U2Net = 320.")]
    public int inputSize = 320;

    [Header("Input")]
    [Tooltip("세그멘테이션할 스캔 이미지 (Texture2D).")]
    public Texture2D inputTexture;
    [Tooltip("시작 시 자동 실행.")]
    public bool autoRunOnStart = false;

    [Header("Parameters")]
    [Range(0f, 1f)]
    [Tooltip("살리언시 이진화 임계값 (0.4~0.6 권장).")]
    public float threshold = 0.5f;
    [Range(0f, 1f)]
    [Tooltip("중앙 가우시안 σ (이미지 폭 대비). 0=비활성화, 0.35=권장.")]
    public float centerSigma = 0.35f;
    [Tooltip("모폴로지 열림(Open) 커널 크기. 경계 돌출 픽셀 제거. 0=비활성화.")]
    [Range(0, 21)]
    public int edgeOpenSize = 5;

    [Header("Output")]
    [Tooltip("결과 마스크를 표시할 RawImage (선택).")]
    public UnityEngine.UI.RawImage outputView;
    [Tooltip("마지막으로 생성된 결과 텍스처 (읽기 전용).")]
    public Texture2D lastResult;

    // ── 내부 ──────────────────────────────────────────────────
    private Worker _worker;

    // ─────────────────────────────────────────────────────────
    void Start()
    {
        InitWorker();
        if (autoRunOnStart && inputTexture != null)
            RunSegmentation();
    }

    void OnDestroy()
    {
        _worker?.Dispose();
    }

    // ─────────────────────────────────────────────────────────
    void InitWorker()
    {
        if (segModel == null)
        {
            Debug.LogError("[FishSentis] segModel이 할당되지 않았습니다.");
            return;
        }
        _worker?.Dispose();
        var model = ModelLoader.Load(segModel);
        _worker = new Worker(model, backend);
        Debug.Log($"[FishSentis] Worker 초기화 완료 (backend={backend}, inputSize={inputSize})");
    }

    // Inspector 버튼 대용 — Context Menu에서 호출 가능
    [ContextMenu("Run Segmentation")]
    public void RunSegmentation()
    {
        if (_worker == null) { Debug.LogError("[FishSentis] Worker 없음. segModel을 확인하세요."); return; }
        if (inputTexture == null) { Debug.LogError("[FishSentis] inputTexture가 없습니다."); return; }

        using Mat rgba = Tex2DToMat(inputTexture);
        using Mat result = Segment(rgba);

        // 결과 → Texture2D
        if (lastResult != null) Destroy(lastResult);
        lastResult = MatToTex2D(result, inputTexture.width, inputTexture.height);
        if (outputView != null) outputView.texture = lastResult;
        Debug.Log($"[FishSentis] 완료: {inputTexture.width}×{inputTexture.height}");
    }

    // ── 세그멘테이션 핵심 로직 ────────────────────────────────
    Mat Segment(Mat rgba)
    {
        int W = rgba.Cols, H = rgba.Rows, sz = inputSize;

        // ① 전처리
        float[] inputData = Preprocess(rgba, sz);
        using var inputTensor = new Tensor<float>(new TensorShape(1, 3, sz, sz), inputData);

        // ② 추론
        _worker.Schedule(inputTensor);
        using var rawTensor = _worker.PeekOutput() as Tensor<float>;
        if (rawTensor == null) { Debug.LogError("[FishSentis] 출력 텐서 취득 실패"); return Mat.Zeros(H, W, MatType.CV_8UC1); }
        // Sentis 2.x: GPU 텐서 → CPU 복사본으로 변환해야 인덱서 접근 가능
        using var salTensor = rawTensor.ReadbackAndClone();

        Debug.Log($"[FishSentis] 추론 완료  출력shape={salTensor.shape}");

        // ③ 살리언시 맵 → Mat + 중앙 가우시안
        using Mat salMat = SaliencyToMat(salTensor, sz, sz);

        // ④ 원본 크기 리사이즈
        using Mat salResized = new Mat();
        Cv2.Resize(salMat, salResized, new OpenCvSharp.Size(W, H));

        // ⑤ 이진화
        Mat alphaMask = new Mat();
        Cv2.Threshold(salResized, alphaMask, (int)(threshold * 255), 255, ThresholdTypes.Binary);
        Debug.Log($"[FishSentis] 이진화 threshold={threshold:F2}  " +
                  $"마스크={Cv2.CountNonZero(alphaMask)}px / {W * H}px");

        // ⑥ 엣지 Open (선택)
        if (edgeOpenSize >= 3)
        {
            int ks = edgeOpenSize % 2 == 0 ? edgeOpenSize + 1 : edgeOpenSize;
            using Mat k = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(ks, ks));
            Cv2.MorphologyEx(alphaMask, alphaMask, MorphTypes.Open, k);
        }

        // ⑦ RGBA에 알파 적용 후 반환
        using Mat rgbaCopy = rgba.Clone();
        Cv2.MixChannels(new[] { alphaMask }, new[] { rgbaCopy }, new[] { 0, 3 });

        // 반환: 알파가 적용된 RGBA Mat
        Mat output = rgbaCopy.Clone();
        alphaMask.Dispose();
        return output;
    }

    // ── 전처리: Texture2D → float[1,3,sz,sz] ImageNet 정규화 ──
    static float[] Preprocess(Mat rgba, int sz)
    {
        using Mat rgb = new Mat();
        Cv2.CvtColor(rgba, rgb, ColorConversionCodes.RGBA2RGB);
        using Mat resized = new Mat();
        Cv2.Resize(rgb, resized, new OpenCvSharp.Size(sz, sz));

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

    // ── 살리언시 맵 → CV_8UC1 (중앙 가우시안 가중치) ──────────
    Mat SaliencyToMat(Tensor<float> tensor, int W, int H)
    {
        float cx = W * 0.5f, cy = H * 0.5f;
        float sigX = centerSigma * W, sigY = centerSigma * H;
        bool  useGauss = centerSigma > 0.01f;

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
        Debug.Log($"[FishSentis] 살리언시 범위: {salMin:F3}~{salMax:F3}  " +
                  $"가우시안={useGauss}(σ={centerSigma:F2})");
        return mat;
    }

    // ── Texture2D → RGBA Mat ──────────────────────────────────
    static Mat Tex2DToMat(Texture2D tex)
    {
        Color32[] pixels = tex.GetPixels32();
        byte[] data = new byte[pixels.Length * 4];
        for (int i = 0; i < pixels.Length; i++)
        {
            data[i * 4 + 0] = pixels[i].r;
            data[i * 4 + 1] = pixels[i].g;
            data[i * 4 + 2] = pixels[i].b;
            data[i * 4 + 3] = pixels[i].a;
        }
        Mat mat = new Mat(tex.height, tex.width, MatType.CV_8UC4);
        Marshal.Copy(data, 0, mat.Data, data.Length);
        Cv2.Flip(mat, mat, FlipMode.X);
        return mat;
    }

    // ── RGBA Mat → Texture2D (원본 크기로 리사이즈) ───────────
    static Texture2D MatToTex2D(Mat rgba, int targetW, int targetH)
    {
        using Mat resized = new Mat();
        if (rgba.Cols != targetW || rgba.Rows != targetH)
            Cv2.Resize(rgba, resized, new OpenCvSharp.Size(targetW, targetH));
        else
            rgba.CopyTo(resized);

        Cv2.Flip(resized, resized, FlipMode.X);
        byte[] data = new byte[resized.Total() * resized.ElemSize()];
        Marshal.Copy(resized.Data, data, 0, data.Length);

        var tex = new Texture2D(targetW, targetH, TextureFormat.RGBA32, false);
        Color32[] pixels = new Color32[targetW * targetH];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = new Color32(data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]);
        tex.SetPixels32(pixels);
        tex.Apply();
        return tex;
    }
}
#else
// com.unity.sentis 패키지가 설치되지 않은 경우 빈 클래스로 대체
// Package Manager → Add package by name → com.unity.sentis 설치 후 활성화됨
public class FishSentisSegmentation : UnityEngine.MonoBehaviour { }
#endif

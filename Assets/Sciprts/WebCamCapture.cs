using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using OpenCvSharp;
using System.Runtime.InteropServices;

public class WebCamCapture : MonoBehaviour
{
    [Header("Camera Settings")]
    public int deviceIndex = 0;

    [Header("Digital Zoom")]
    [Range(1f, 5f), Tooltip("디지털 줌 배율. 1=원본, 2=2배 확대. 마우스 스크롤로 조정 가능.")]
    public float zoomFactor = 1f;
    [Tooltip("스크롤 1회당 줌 변화량.")]
    public float zoomStep = 0.1f;

    [Header("UI")]
    public RawImage previewImage;

    public WebCamTexture CamTexture { get; private set; }
    public bool IsPlaying => CamTexture != null && CamTexture.isPlaying
                             && CamTexture.width > 16;

    private Color32[] _camPixels;
    private byte[]    _rawBuffer;

    // ── 카메라 시작 ───────────────────────────────────────────
    public void StartCamera()
    {
        var devices = WebCamTexture.devices;
        if (devices.Length == 0) { Debug.LogError("[WebCam] 카메라 없음"); return; }

        // 연결된 카메라 전체 목록 출력 (deviceIndex 잘못 설정 시 확인용)
        for (int i = 0; i < devices.Length; i++)
            GameLog.Log($"[WebCam] 감지된 카메라 [{i}]: {devices[i].name}");

        string camName = devices[Mathf.Clamp(deviceIndex, 0, devices.Length - 1)].name;
        GameLog.Log($"[WebCam] 선택된 카메라 (deviceIndex={deviceIndex}): {camName}");
        StartCoroutine(StartWithFallback(camName));
    }

    // ── Fallback 체인: 해상도 / FPS 조합을 순서대로 시도 ─────
    IEnumerator StartWithFallback(string camName)
    {
        var candidates = new (int w, int h, int fps)[]
        {
            (2560, 1440, 60),       
            (1920, 1080, 60),               // FHD 60fps
            (1920, 1080,  0),               // FHD 자동
            (1280,  720, 60),               // HD 60fps
            (1280,  720,  0),               // HD 자동
            ( 640,  480,  0),               // VGA 자동
        };

        foreach (var (w, h, fps) in candidates)
        {
            // 이전 시도 정리
            if (CamTexture != null)
            {
                CamTexture.Stop();
                Object.Destroy(CamTexture);
                CamTexture = null;
                yield return null;
            }

            CamTexture = fps > 0
                ? new WebCamTexture(camName, w, h, fps)
                : new WebCamTexture(camName, w, h);
            CamTexture.filterMode = FilterMode.Bilinear;
            CamTexture.Play();

            if (previewImage != null)
                previewImage.texture = CamTexture;

            // 최대 3초 대기 후 실제 해상도 확인
            float timeout = 3f;
            while (timeout > 0f && CamTexture.width <= 16)
            {
                timeout -= Time.deltaTime;
                yield return null;
            }

            if (CamTexture.width > 16)
            {
                GameLog.Log($"[WebCam] ✓ 실제: {CamTexture.width}×{CamTexture.height}@{CamTexture.requestedFPS}fps  (요청={w}×{h}@{fps})");
                yield break;
            }

            Debug.LogWarning($"[WebCam] 실패: {w}×{h}@{fps} → 다음 설정 시도");
        }

        Debug.LogError("[WebCam] 모든 해상도 시도 실패.\n" +
                       "• 다른 앱(Razer Synapse, OBS 등)이 카메라를 점유 중인지 확인\n" +
                       "• deviceIndex가 올바른지 확인");
    }

    // ── 최신 프레임 → RGBA Mat 반환 (호출자가 Dispose 책임) ──
    // zoomFactor > 1 이면 중앙 영역을 크롭 후 원본 해상도로 Lanczos4 리사이즈.
    public Mat GrabFrameRGBA()
    {
        if (!IsPlaying) return null;

        int camW   = CamTexture.width;
        int camH   = CamTexture.height;
        int count  = camW * camH;
        int needed = count * 4;

        if (_camPixels == null || _camPixels.Length != count)
            _camPixels = new Color32[count];
        if (_rawBuffer == null || _rawBuffer.Length != needed)
            _rawBuffer = new byte[needed];

        CamTexture.GetPixels32(_camPixels);

        for (int i = 0; i < count; i++)
        {
            _rawBuffer[i * 4    ] = _camPixels[i].r;
            _rawBuffer[i * 4 + 1] = _camPixels[i].g;
            _rawBuffer[i * 4 + 2] = _camPixels[i].b;
            _rawBuffer[i * 4 + 3] = 255;
        }

        Mat mat = new Mat(camH, camW, MatType.CV_8UC4);
        Marshal.Copy(_rawBuffer, 0, mat.Data, needed);
        Cv2.Flip(mat, mat, FlipMode.X);

        // 디지털 줌: 중앙 크롭 후 원본 크기로 확대
        if (zoomFactor > 1.001f)
        {
            try
            {
                float invZoom  = 1f / zoomFactor;
                int   cropW    = Mathf.Max(1, Mathf.RoundToInt(camW * invZoom));
                int   cropH    = Mathf.Max(1, Mathf.RoundToInt(camH * invZoom));
                int   x0       = (camW - cropW) / 2;
                int   y0       = (camH - cropH) / 2;
                var   roi      = new OpenCvSharp.Rect(x0, y0, cropW, cropH);
                using Mat cropped = new Mat(mat, roi);
                Mat zoomed = new Mat();
                Cv2.Resize(cropped, zoomed, new Size(camW, camH), 0, 0, InterpolationFlags.Lanczos4);
                return zoomed;
            }
            finally
            {
                mat.Dispose();
            }
        }

        return mat;
    }

    // ── 줌 조작 (스크롤 휠) ──────────────────────────────────
    void Update()
    {
        float scroll = Input.mouseScrollDelta.y;
        if (Mathf.Abs(scroll) > 0f)
        {
            zoomFactor = Mathf.Clamp(zoomFactor - scroll * zoomStep, 1f, 5f);

            if (previewImage != null)
            {
                float size   = 1f / zoomFactor;
                float offset = (1f - size) * 0.5f;
                previewImage.uvRect = new UnityEngine.Rect(offset, offset, size, size);
            }
        }
    }

    void OnDestroy()
    {
        if (CamTexture != null)
        {
            CamTexture.Stop();
            Object.Destroy(CamTexture);
            CamTexture = null;
        }
    }
}

using UnityEngine;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using OpenCvSharp;
using OpenCvSharp.Aruco;
using ArucoDict = OpenCvSharp.Aruco.Dictionary; // Dictionary 이름 충돌 방지

public class ArucoDetector : MonoBehaviour
{
    [Header("Settings")]
    public int outputWidth = 3508;
    public int outputHeight = 2480;
    public float holdDurationToCapture = 1.5f;
    [Header("Debug UI")]
    public UnityEngine.UI.RawImage overlayView;
    public UnityEngine.UI.Text failureText;

    public event Action<Texture2D, int> OnAllMarkersDetected;

    private WebCamCapture cam;
    private ArucoDict arucoDict;
    private DetectorParameters detParams;
    private float holdTimer = 0f;

    // ── Fish 마스크 (동적) ──────────────────────────────────────
    private int fishCount = 0;
    private Texture2D[] fishMaskSources;
    private Mat[] _cachedFishMasks;
    private int _cachedMaskW = 0;
    private int _cachedMaskH = 0;

    // ── 재사용 버퍼 (매 프레임 GC 방지) ──────────────────────
    private byte[] _rawBuffer;
    private readonly Dictionary<int, int> _idToIdx = new Dictionary<int, int>();

    // ── 회전 레이블 (상수) ────────────────────────────────────
    private static readonly string[] s_rotLabels = { "회전 없음", "90°CCW", "180°", "90°CW" };

    // ── 초기화 ────────────────────────────────────────────────
    void Start()
    {
        cam = ScannerManager.Instance.webCam;
        arucoDict = CvAruco.GetPredefinedDictionary(PredefinedDictionaryName.Dict4X4_50);
        detParams = DetectorParameters.Create();
        SetFailureMessage(null);

        // Resources/Fish/fish0, fish1, ... 순서로 로드
        var list = new List<Texture2D>();
        for (int i = 0; ; i++)
        {
            var tex = Resources.Load<Texture2D>($"Fish/fish{i}");
            if (tex == null) break;
            list.Add(tex);
        }

        fishCount = list.Count;
        if (fishCount == 0)
        {
            Debug.LogWarning("[ArucoDetector] Resources/Fish 에 fish*.png 파일이 없습니다.");
            fishMaskSources = Array.Empty<Texture2D>();
            _cachedFishMasks = Array.Empty<Mat>();
            return;
        }

        fishMaskSources = list.ToArray();
        _cachedFishMasks = new Mat[fishCount];
        Debug.Log($"[ArucoDetector] Fish {fishCount}개 로드 → 인식 ID 범위 0~{fishCount * 4 - 1}");
    }

    // ── 종료 시 UI 텍스처·캐시 정리 ─────────────────────────
    void OnDestroy()
    {
        if (overlayView != null && overlayView.texture != null)
            Destroy(overlayView.texture);

        if (_cachedFishMasks != null)
            foreach (var m in _cachedFishMasks)
                m?.Dispose();
    }

    // ── 매 프레임 감지 ────────────────────────────────────────
    void Update()
    {
        if (ScannerManager.Instance.State != ScannerState.Scanning) return;
        if (!cam.CamTexture.isPlaying) return;

        using Mat frame = WebCamToMat(cam.CamTexture);
        using Mat gray = new Mat();
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

    // ── 실패 안내 문구 표시/숨김 ─────────────────────────────
    void SetFailureMessage(string message)
    {
        if (failureText == null) return;
        bool hasMessage = !string.IsNullOrEmpty(message);
        failureText.text = hasMessage ? message : string.Empty;
        failureText.gameObject.SetActive(hasMessage);
    }

    // ── WebCamTexture → RGBA Mat ──────────────────────────────
    // alpha 강제 255, Unity Y축 반전 보정 / _rawBuffer 재사용으로 GC 최소화
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
    // Unity 표시용 Y축 복원 (반환된 Texture2D는 호출자가 Destroy 책임)
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
    // fishCount 그룹(각 4개 ID) 중 완전한 그룹을 발견하면 true 반환
    (bool, Point2f[][], int[]) TryDetect(Mat gray)
    {
        CvAruco.DetectMarkers(
            gray, arucoDict,
            out Point2f[][] allCorners,
            out int[] allIds,
            detParams,
            out _);

        if (allIds == null || allIds.Length < 4)
            return (false, null, null);

        // _idToIdx 재사용 (매 프레임 Dictionary 신규 할당 방지)
        _idToIdx.Clear();
        for (int i = 0; i < allIds.Length; i++)
            _idToIdx[allIds[i]] = i;

        // 그룹(fish0=0~3, fish1=4~7, ...) 순서대로 완전한 4개 체크
        for (int g = 0; g < fishCount; g++)
        {
            int b = g * 4;
            if (!_idToIdx.ContainsKey(b) || !_idToIdx.ContainsKey(b + 1) ||
                !_idToIdx.ContainsKey(b + 2) || !_idToIdx.ContainsKey(b + 3))
                continue;

            return (true,
                new Point2f[4][]
                {
                    allCorners[_idToIdx[b    ]],
                    allCorners[_idToIdx[b + 1]],
                    allCorners[_idToIdx[b + 2]],
                    allCorners[_idToIdx[b + 3]]
                },
                new int[] { b, b + 1, b + 2, b + 3 });
        }

        return (false, null, null);
    }

    // ── 공통: 마커 센터 배열 + 중점 계산 ────────────────────
    static (Point2f[] centers, float midX, float midY) ComputeCenters(Point2f[][] corners, int count)
    {
        var centers = new Point2f[count];
        float sumX = 0, sumY = 0;
        for (int i = 0; i < count; i++)
        {
            var c = corners[i];
            centers[i] = new Point2f(
                (c[0].X + c[1].X + c[2].X + c[3].X) / 4f,
                (c[0].Y + c[1].Y + c[2].Y + c[3].Y) / 4f);
            sumX += centers[i].X;
            sumY += centers[i].Y;
        }
        return (centers, sumX / count, sumY / count);
    }

    // ── 마커 내부 꼭짓점(중심에 가장 가까운 모서리) TL/TR/BR/BL ──
    static Point2f[] GetInnerCorners(Point2f[][] corners, int[] ids)
    {
        var (centers, midX, midY) = ComputeCenters(corners, ids.Length);
        Point2f tl = default, tr = default, br = default, bl = default;
        for (int i = 0; i < ids.Length; i++)
        {
            Point2f inner = ClosestCorner(corners[i], midX, midY);
            if      (centers[i].X <= midX && centers[i].Y <= midY) tl = inner;
            else if (centers[i].X >  midX && centers[i].Y <= midY) tr = inner;
            else if (centers[i].X >  midX && centers[i].Y >  midY) br = inner;
            else                                                    bl = inner;
        }
        return new[] { tl, tr, br, bl };
    }

    static Point2f ClosestCorner(Point2f[] c, float tx, float ty)
    {
        Point2f best = c[0];
        float minD = float.MaxValue;
        foreach (var p in c)
        {
            float d = (p.X - tx) * (p.X - tx) + (p.Y - ty) * (p.Y - ty);
            if (d < minD) { minD = d; best = p; }
        }
        return best;
    }

    // ── 마커 외부 꼭짓점(중심에서 가장 먼 모서리) TL/TR/BR/BL ──
    static Point2f[] GetOuterCorners(Point2f[][] corners, int[] ids)
    {
        var (centers, midX, midY) = ComputeCenters(corners, ids.Length);
        Point2f tl = default, tr = default, br = default, bl = default;
        for (int i = 0; i < ids.Length; i++)
        {
            Point2f outer = FarthestCorner(corners[i], midX, midY);
            if      (centers[i].X <= midX && centers[i].Y <= midY) tl = outer;
            else if (centers[i].X >  midX && centers[i].Y <= midY) tr = outer;
            else if (centers[i].X >  midX && centers[i].Y >  midY) br = outer;
            else                                                    bl = outer;
        }
        return new[] { tl, tr, br, bl };
    }

    static Point2f FarthestCorner(Point2f[] c, float tx, float ty)
    {
        Point2f best = c[0];
        float maxD = 0;
        foreach (var p in c)
        {
            float d = (p.X - tx) * (p.X - tx) + (p.Y - ty) * (p.Y - ty);
            if (d > maxD) { maxD = d; best = p; }
        }
        return best;
    }

    // ── TL 위치 마커의 ID 반환 ────────────────────────────────
    static int FindTLMarkerId(Point2f[][] corners, int[] ids)
    {
        var (centers, midX, midY) = ComputeCenters(corners, ids.Length);
        for (int i = 0; i < ids.Length; i++)
            if (centers[i].X <= midX && centers[i].Y <= midY)
                return ids[i];
        return ids[0]; // fallback
    }

    // ── ID%4==0 마커 위치 → 회전 오프셋 반환 ────────────────
    // 0=TL(무회전) / 1=TR(90°CCW) / 2=BR(180°) / 3=BL(90°CW)
    static int FindAnchorOffset(Point2f[][] corners, int[] ids)
    {
        var (centers, midX, midY) = ComputeCenters(corners, ids.Length);
        for (int i = 0; i < ids.Length; i++)
        {
            if (ids[i] % 4 != 0) continue;
            bool left = centers[i].X <= midX;
            bool top  = centers[i].Y <= midY;
            if ( left &&  top) return 0; // TL → 무회전
            if (!left &&  top) return 1; // TR → 90° CCW
            if (!left && !top) return 2; // BR → 180°
            return 3;                    // BL → 90° CW
        }
        return 0; // fallback
    }

    // ── 퍼스펙티브 워프 → 회전 보정 → Fish 마스크 → crop ──────
    Texture2D Warp(Mat frame, Point2f[][] corners, int[] ids, int tlMarkerId)
    {
        int fishIndex = tlMarkerId / 4;

        Point2f[] src = GetOuterCorners(corners, ids);
        int warpW = outputWidth;
        int warpH = outputHeight;

        Point2f[] dst = {
            new Point2f(0,     0),
            new Point2f(warpW, 0),
            new Point2f(warpW, warpH),
            new Point2f(0,     warpH)
        };

        using Mat M = Cv2.GetPerspectiveTransform(src, dst);
        using Mat warped = new Mat();
        Cv2.WarpPerspective(frame, warped, M, new OpenCvSharp.Size(warpW, warpH));

        // 내부 꼭짓점을 워프 공간으로 변환 → 마커 픽셀 크기(crop 여백) 계산
        Point2f[] innerWarped = TransformPoints(M, GetInnerCorners(corners, ids));
        int cropX = Mathf.Max(0, Mathf.RoundToInt(innerWarped[0].X));
        int cropY = Mathf.Max(0, Mathf.RoundToInt(innerWarped[0].Y));
        Debug.Log($"[ArucoDetector] fish{fishIndex} 마스크, 마커 crop: x={cropX}px, y={cropY}px");

        int anchorPos = FindAnchorOffset(corners, ids);

        if (anchorPos == 0)
            return CropAndMask(warped, s_rotLabels[0], cropX, cropY, fishIndex);

        RotateFlags flag = anchorPos == 1 ? (RotateFlags)2
                         : anchorPos == 2 ? RotateFlags.Rotate180
                         :                  RotateFlags.Rotate90Clockwise;
        using Mat rotated = new Mat();
        Cv2.Rotate(warped, rotated, flag);

        // 90°/270° 회전 시 가로↔세로 교환 → warpW×warpH 리사이즈, crop축 교환
        if (anchorPos == 1 || anchorPos == 3)
        {
            int cropXRot = Mathf.RoundToInt(cropY * (float)warpW / warpH);
            int cropYRot = Mathf.RoundToInt(cropX * (float)warpH / warpW);
            using Mat resized = new Mat();
            Cv2.Resize(rotated, resized, new OpenCvSharp.Size(warpW, warpH));
            return CropAndMask(resized, s_rotLabels[anchorPos], cropXRot, cropYRot, fishIndex);
        }

        return CropAndMask(rotated, s_rotLabels[anchorPos], cropX, cropY, fishIndex);
    }

    Texture2D CropAndMask(Mat mat, string rotLabel, int cropX, int cropY, int fishIndex)
    {
        ApplyFishMask(mat, fishIndex);

        if (cropX > 0 || cropY > 0)
        {
            int cropW = mat.Cols - 2 * cropX;
            int cropH = mat.Rows - 2 * cropY;
            if (cropW > 0 && cropH > 0)
            {
                using Mat cropped = new Mat(mat, new OpenCvSharp.Rect(cropX, cropY, cropW, cropH));
                Debug.Log($"[ArucoDetector] 최종 출력: {cropW}×{cropH}px ({rotLabel}, crop x:{cropX} y:{cropY})");
                return MatToTexture2D(cropped);
            }
        }

        Debug.Log($"[ArucoDetector] 최종 출력: {mat.Cols}×{mat.Rows}px ({rotLabel})");
        return MatToTexture2D(mat);
    }

    // ── 퍼스펙티브 변환 행렬로 점 배열 변환 ──────────────────
    static Point2f[] TransformPoints(Mat H, Point2f[] pts)
    {
        using Mat src = new Mat(pts.Length, 1, MatType.CV_32FC2);
        for (int i = 0; i < pts.Length; i++)
            src.Set<Vec2f>(i, 0, new Vec2f(pts[i].X, pts[i].Y));

        using Mat dst = new Mat();
        Cv2.PerspectiveTransform(src, dst, H);

        var result = new Point2f[pts.Length];
        for (int i = 0; i < pts.Length; i++)
        {
            Vec2f v = dst.At<Vec2f>(i, 0);
            result[i] = new Point2f(v.Item0, v.Item1);
        }
        return result;
    }

    // ── fish{fishIndex} 마스크 적용 ───────────────────────────
    void ApplyFishMask(Mat rgba, int fishIndex)
    {
        if (fishIndex < 0 || fishIndex >= fishCount) return;
        if (fishMaskSources[fishIndex] == null) return;

        int w = rgba.Cols, h = rgba.Rows;

        // 출력 크기가 바뀌면 모든 캐시 무효화
        if (_cachedMaskW != w || _cachedMaskH != h)
        {
            foreach (var m in _cachedFishMasks) m?.Dispose();
            _cachedFishMasks = new Mat[fishCount];
            _cachedMaskW = w;
            _cachedMaskH = h;
        }

        _cachedFishMasks[fishIndex] ??= BuildFishMask(fishMaskSources[fishIndex], w, h);

        if (_cachedFishMasks[fishIndex] != null && !_cachedFishMasks[fishIndex].Empty())
        {
            Cv2.MixChannels(new[] { _cachedFishMasks[fishIndex] }, new[] { rgba }, new[] { 0, 3 });
            Debug.Log($"[ArucoDetector] Fish{fishIndex} 마스크 적용 완료.");
        }
    }

    // ── fish{N}.png → alpha 마스크 생성 ──────────────────────
    // alpha 채널만 1채널로 직접 복사 (4채널 복사 대비 메모리 1/4)
    static Mat BuildFishMask(Texture2D source, int targetW, int targetH)
    {
        Color32[] pixels = source.GetPixels32();
        byte[] alpha = new byte[pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
            alpha[i] = pixels[i].a;

        using Mat alphaMat = new Mat(source.height, source.width, MatType.CV_8UC1);
        Marshal.Copy(alpha, 0, alphaMat.Data, alpha.Length);
        Cv2.Flip(alphaMat, alphaMat, FlipMode.X); // Unity Y축 반전 복원

        Mat mask = new Mat();
        Cv2.Resize(alphaMat, mask, new OpenCvSharp.Size(targetW, targetH));
        Debug.Log($"[ArucoDetector] Fish 마스크 생성: {targetW}×{targetH}px");
        return mask;
    }

    // ── 마커 내부 영역 크기 디버그 출력 ──────────────────────
    public void LogInnerAreaSize(Point2f[][] corners, int[] ids)
    {
        Point2f[] inner = GetInnerCorners(corners, ids);
        Point2f tl = inner[0], tr = inner[1], br = inner[2], bl = inner[3];

        float topW    = Mathf.Sqrt((tr.X - tl.X) * (tr.X - tl.X) + (tr.Y - tl.Y) * (tr.Y - tl.Y));
        float bottomW = Mathf.Sqrt((br.X - bl.X) * (br.X - bl.X) + (br.Y - bl.Y) * (br.Y - bl.Y));
        float leftH   = Mathf.Sqrt((bl.X - tl.X) * (bl.X - tl.X) + (bl.Y - tl.Y) * (bl.Y - tl.Y));
        float rightH  = Mathf.Sqrt((br.X - tr.X) * (br.X - tr.X) + (br.Y - tr.Y) * (br.Y - tr.Y));

        Debug.Log($"[ArucoDetector] 마커 내부 영역 크기\n" +
                  $"  Width  → 상단: {topW:F1}px / 하단: {bottomW:F1}px / 평균: {(topW+bottomW)/2f:F1}px\n" +
                  $"  Height → 좌측: {leftH:F1}px / 우측: {rightH:F1}px / 평균: {(leftH+rightH)/2f:F1}px\n" +
                  $"  TL({tl.X:F0},{tl.Y:F0})  TR({tr.X:F0},{tr.Y:F0})\n" +
                  $"  BL({bl.X:F0},{bl.Y:F0})  BR({br.X:F0},{br.Y:F0})");
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

        if (overlayView.texture != null)
            Destroy(overlayView.texture);
        overlayView.texture = MatToTexture2D(overlayRGBA);
    }
}

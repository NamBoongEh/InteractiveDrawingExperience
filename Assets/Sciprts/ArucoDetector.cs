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
    public int outputWidth = 1920;
    public int outputHeight = 1080;
    public int[] targetIds = { 0, 1, 2, 3 };
    public float holdDurationToCapture = 1.5f;
    [Range(100, 254)] public int backgroundThreshold = 200;

    [Header("Debug UI")]
    public UnityEngine.UI.RawImage overlayView;
    public UnityEngine.UI.Text failureText; // 생성 실패 시 안내 문구

    [Header("Fish Mask Settings")]
    public Texture2D fishMaskSource;                  // Inspector에서 Fish/fish0 할당
    [Tooltip("fish0.png 내 마커 1개의 픽셀 크기 — 내부영역 = (파일크기 - 2×마커크기)")]
    public int fishSourceMarkerPx = 378;              // 내부: 3390-756=2634, 2362-756=1606

    public event Action<Texture2D, int> OnAllMarkersDetected;

    private WebCamCapture cam;
    private ArucoDict arucoDict;
    private DetectorParameters detParams;
    private float holdTimer = 0f;
    private Mat _cachedFishMask;                      // BuildFishMask() 결과 캐시

    // ── 초기화 ────────────────────────────────────────────────
    void Start()
    {
        cam = ScannerManager.Instance.webCam;
        arucoDict = CvAruco.GetPredefinedDictionary(PredefinedDictionaryName.Dict4X4_50);
        detParams = DetectorParameters.Create();
        SetFailureMessage(null);

        // Inspector 미할당 시 Resources에서 자동 로드
        if (fishMaskSource == null)
        {
            fishMaskSource = Resources.Load<Texture2D>("Fish/fish0");
            if (fishMaskSource == null)
                Debug.LogWarning("[ArucoDetector] Fish/fish0 텍스처를 Resources에서 찾을 수 없습니다. fallback 마스킹 사용.");
            else
                Debug.Log("[ArucoDetector] Fish/fish0 텍스처 자동 로드 완료.");
        }
    }

    // ── 종료 시 UI 텍스처·캐시 정리 ─────────────────────────
    void OnDestroy()
    {
        if (overlayView != null && overlayView.texture != null)
            Destroy(overlayView.texture);
        _cachedFishMask?.Dispose();
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

            // holdTimer가 막 시작될 때 한 번만 크기 출력
            if (holdTimer == 0f)
                LogInnerAreaSize(corners, ids);

            holdTimer += Time.deltaTime;

            if (holdTimer >= holdDurationToCapture)
            {
                holdTimer = 0f;

                try
                {
                    int tlMarkerId = FindTLMarkerId(corners, ids);
                    Texture2D warped = Warp(frame, corners, ids);
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
    // alpha 강제 255 (웹캠은 투명도 없음), Unity Y축 반전 보정
    Mat WebCamToMat(WebCamTexture webcam)
    {
        Color32[] pixels = webcam.GetPixels32();
        byte[] raw = new byte[pixels.Length * 4];

        for (int i = 0; i < pixels.Length; i++)
        {
            raw[i * 4    ] = pixels[i].r;
            raw[i * 4 + 1] = pixels[i].g;
            raw[i * 4 + 2] = pixels[i].b;
            raw[i * 4 + 3] = 255;
        }

        Mat mat = new Mat(webcam.height, webcam.width, MatType.CV_8UC4);
        Marshal.Copy(raw, 0, mat.Data, raw.Length);
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
    // targetIds 전부 발견한 경우에만 true 반환
    (bool, Point2f[][], int[]) TryDetect(Mat gray)
    {
        CvAruco.DetectMarkers(
            gray, arucoDict,
            out Point2f[][] corners,
            out int[] ids,
            detParams,
            out _);

        if (ids == null || ids.Length < targetIds.Length)
            return (false, null, null);

        var foundSet = new HashSet<int>(ids);
        foreach (int t in targetIds)
            if (!foundSet.Contains(t)) return (false, null, null);

        return (true, corners, ids);
    }

    // ── 마커 내부 꼭짓점(종이 중심에 가장 가까운 모서리) TL/TR/BR/BL 반환 ──
    static Point2f[] GetInnerCorners(Point2f[][] corners, int[] ids)
    {
        float sumX = 0, sumY = 0;
        var centers = new Point2f[ids.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            var c = corners[i];
            centers[i] = new Point2f(
                (c[0].X + c[1].X + c[2].X + c[3].X) / 4f,
                (c[0].Y + c[1].Y + c[2].Y + c[3].Y) / 4f);
            sumX += centers[i].X;
            sumY += centers[i].Y;
        }
        float midX = sumX / ids.Length;
        float midY = sumY / ids.Length;

        Point2f tl = default, tr = default, br = default, bl = default;
        for (int i = 0; i < ids.Length; i++)
        {
            Point2f inner = ClosestCorner(corners[i], midX, midY);
            Point2f center = centers[i];

            if      (center.X <= midX && center.Y <= midY) tl = inner;
            else if (center.X >  midX && center.Y <= midY) tr = inner;
            else if (center.X >  midX && center.Y >  midY) br = inner;
            else                                            bl = inner;
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

    // ── TL 위치 마커의 ID 반환 ────────────────────────────────
    static int FindTLMarkerId(Point2f[][] corners, int[] ids)
    {
        float sumX = 0, sumY = 0;
        var centers = new Point2f[ids.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            var c = corners[i];
            centers[i] = new Point2f(
                (c[0].X + c[1].X + c[2].X + c[3].X) / 4f,
                (c[0].Y + c[1].Y + c[2].Y + c[3].Y) / 4f);
            sumX += centers[i].X;
            sumY += centers[i].Y;
        }
        float midX = sumX / ids.Length;
        float midY = sumY / ids.Length;

        for (int i = 0; i < ids.Length; i++)
            if (centers[i].X <= midX && centers[i].Y <= midY)
                return ids[i];

        return ids[0]; // fallback
    }

    // ── 퍼스펙티브 워프 → 회전 보정 → Fish0 마스크 ──
    // 처리 순서: 4점 퍼스펙티브 변환 → 회전(ID%4==0 좌상단) → Fish0 마스크(배경 투명화 포함)
    // warpedInner = 실제 감지된 내부꼭짓점 위치 → Fish0 동적 정렬로 그림·마스크 일치 보장
    // 반환된 Texture2D는 호출자가 Destroy() 책임
    Texture2D Warp(Mat frame, Point2f[][] corners, int[] ids)
    {
        Point2f[] src = GetInnerCorners(corners, ids);

        // ── 1920×1080 강제 전 자연 크기 계산 및 로그 ──────────
        Point2f tl = src[0], tr = src[1], br = src[2], bl = src[3];
        float naturalW = ((tr.X - tl.X + (br.X - bl.X)) / 2f);   // 상·하단 가로 평균
        float naturalH = ((bl.Y - tl.Y + (br.Y - tr.Y)) / 2f);   // 좌·우측 세로 평균
        // 투시 왜곡을 고려한 실제 거리(유클리드)
        float topW    = Mathf.Sqrt((tr.X-tl.X)*(tr.X-tl.X) + (tr.Y-tl.Y)*(tr.Y-tl.Y));
        float bottomW = Mathf.Sqrt((br.X-bl.X)*(br.X-bl.X) + (br.Y-bl.Y)*(br.Y-bl.Y));
        float leftH   = Mathf.Sqrt((bl.X-tl.X)*(bl.X-tl.X) + (bl.Y-tl.Y)*(bl.Y-tl.Y));
        float rightH  = Mathf.Sqrt((br.X-tr.X)*(br.X-tr.X) + (br.Y-tr.Y)*(br.Y-tr.Y));
        float avgW    = (topW + bottomW) / 2f;
        float avgH    = (leftH + rightH) / 2f;
        Debug.Log($"[ArucoDetector] Warp 전 자연 크기 (카메라 원본 기준)\n" +
                  $"  Width  → 상단: {topW:F1}px / 하단: {bottomW:F1}px / 평균: {avgW:F1}px\n" +
                  $"  Height → 좌측: {leftH:F1}px / 우측: {rightH:F1}px / 평균: {avgH:F1}px");

        // 워프 출력 크기: fish0 내부영역(마커 안쪽)과 동일한 픽셀 수로 생성
        // → 마스크와 1:1 대응하여 크기 불일치 방지
        int warpW = outputWidth;
        int warpH = outputHeight;
        if (fishMaskSource != null && fishSourceMarkerPx > 0)
        {
            int iw2 = fishMaskSource.width  - 2 * fishSourceMarkerPx;
            int ih2 = fishMaskSource.height - 2 * fishSourceMarkerPx;
            if (iw2 > 0 && ih2 > 0)
            {
                warpW = iw2;
                warpH = ih2;
                Debug.Log($"[ArucoDetector] 워프 출력 크기: fish0 내부영역 {warpW}×{warpH}px");
            }
        }

        Point2f[] dst = {
            new Point2f(0,      0),
            new Point2f(warpW,  0),
            new Point2f(warpW,  warpH),
            new Point2f(0,      warpH)
        };

        using Mat M = Cv2.GetPerspectiveTransform(src, dst);
        using Mat warped = new Mat();
        Cv2.WarpPerspective(frame, warped, M,
            new OpenCvSharp.Size(warpW, warpH));

        // 마스크는 회전 완료 후 적용 (회전 전 적용 시 alpha 채널도 함께 회전되어 방향 불일치)
        int anchorPos = FindAnchorOffset(corners, ids);
        if (anchorPos == 0)
        {
            ApplyReferenceMask(warped);
            Debug.Log($"[ArucoDetector] 최종 출력: {warped.Cols}×{warped.Rows}px (회전 없음)");
            return MatToTexture2D(warped);
        }

        // RotateFlags: Rotate90Clockwise=0, Rotate180=1, Rotate90Counterclockwise=2
        RotateFlags flag = anchorPos == 1 ? (RotateFlags)2
                         : anchorPos == 2 ? RotateFlags.Rotate180
                         :                  RotateFlags.Rotate90Clockwise;
        string[] rotLabels = { "", "90°CCW", "180°", "90°CW" };
        using Mat rotated = new Mat();
        Cv2.Rotate(warped, rotated, flag);

        // 90°/270° 회전 시 가로↔세로 교환 → warpW×warpH로 리사이즈 후 마스크 적용
        if (anchorPos == 1 || anchorPos == 3)
        {
            Debug.Log($"[ArucoDetector] 회전({rotLabels[anchorPos]}) 후 리사이즈 전: " +
                      $"{rotated.Cols}×{rotated.Rows}px → 최종: {warpW}×{warpH}px");
            using Mat resized = new Mat();
            Cv2.Resize(rotated, resized, new OpenCvSharp.Size(warpW, warpH));
            ApplyReferenceMask(resized);
            return MatToTexture2D(resized);
        }

        ApplyReferenceMask(rotated);
        Debug.Log($"[ArucoDetector] 최종 출력: {rotated.Cols}×{rotated.Rows}px ({rotLabels[anchorPos]} 회전)");
        return MatToTexture2D(rotated);
    }

    // ── ID%4==0 마커 위치 → 회전 오프셋 반환 ────────────────
    // 0=TL(무회전) / 1=TR(90°CCW) / 2=BR(180°) / 3=BL(90°CW)
    static int FindAnchorOffset(Point2f[][] corners, int[] ids)
    {
        float sumX = 0, sumY = 0;
        var centers = new Point2f[ids.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            var c = corners[i];
            centers[i] = new Point2f(
                (c[0].X + c[1].X + c[2].X + c[3].X) / 4f,
                (c[0].Y + c[1].Y + c[2].Y + c[3].Y) / 4f);
            sumX += centers[i].X;
            sumY += centers[i].Y;
        }
        float midX = sumX / ids.Length;
        float midY = sumY / ids.Length;

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

    // ── fish0 기반 마스크 적용 (fallback: 스캔 자체 실루엣) ────
    // fish0.png 마커 내부 영역을 워프 출력 크기로 스케일 → 실루엣 마스크 생성
    void ApplyReferenceMask(Mat rgba)
    {
        // fish0 마스크 캐시 구축 (최초 1회)
        if (fishMaskSource != null && (_cachedFishMask == null || _cachedFishMask.Empty()))
            _cachedFishMask = BuildFishMask(rgba.Cols, rgba.Rows);

        if (_cachedFishMask != null && !_cachedFishMask.Empty())
        {
            Cv2.MixChannels(new[] { _cachedFishMask }, new[] { rgba }, new[] { 0, 3 });
            return;
        }

        // ── fallback: 스캔 이미지 자체에서 실루엣 추출 ──────
        const int scale = 4;
        int sw = rgba.Cols / scale;
        int sh = rgba.Rows / scale;

        using Mat small = new Mat();
        Cv2.Resize(rgba, small, new OpenCvSharp.Size(sw, sh));

        using Mat gray = new Mat();
        Cv2.CvtColor(small, gray, ColorConversionCodes.RGBA2GRAY);
        using Mat binary = new Mat();
        Cv2.Threshold(gray, binary, backgroundThreshold, 255, ThresholdTypes.BinaryInv);

        using Mat kernel = Cv2.GetStructuringElement(
            MorphShapes.Ellipse, new OpenCvSharp.Size(7, 7));
        Cv2.MorphologyEx(binary, binary, MorphTypes.Close, kernel);

        Cv2.FindContours(binary, out Point[][] contours, out _,
            RetrievalModes.External, ContourApproximationModes.ApproxSimple);

        using Mat silSmall = new Mat(sh, sw, MatType.CV_8UC1, new Scalar(0));
        double minArea = sw * sh * 0.001;
        for (int i = 0; i < contours.Length; i++)
            if (Cv2.ContourArea(contours[i]) >= minArea)
                Cv2.DrawContours(silSmall, contours, i, new Scalar(255), thickness: -1);

        using Mat silhouette = new Mat();
        Cv2.Resize(silSmall, silhouette,
            new OpenCvSharp.Size(rgba.Cols, rgba.Rows),
            interpolation: InterpolationFlags.Nearest);

        Cv2.MixChannels(new[] { silhouette }, new[] { rgba }, new[] { 0, 3 });
    }

    // ── fish0.png → 워프 출력 크기 실루엣 마스크 생성 ────────
    // 마커 내부 영역(fishSourceMarkerPx 기준)만 크롭 후 targetW×targetH로 리사이즈
    // 반환된 Mat은 _cachedFishMask에 저장되며 OnDestroy에서 Dispose됨
    Mat BuildFishMask(int targetW, int targetH)
    {
        int fw = fishMaskSource.width;
        int fh = fishMaskSource.height;

        // Texture2D → RGBA Mat (GetPixels32: 플랫폼 독립적 RGBA)
        Color32[] pixels = fishMaskSource.GetPixels32();
        byte[] raw = new byte[pixels.Length * 4];
        for (int i = 0; i < pixels.Length; i++)
        {
            raw[i * 4    ] = pixels[i].r;
            raw[i * 4 + 1] = pixels[i].g;
            raw[i * 4 + 2] = pixels[i].b;
            raw[i * 4 + 3] = pixels[i].a;
        }
        using Mat fishMat = new Mat(fh, fw, MatType.CV_8UC4);
        Marshal.Copy(raw, 0, fishMat.Data, raw.Length);
        Cv2.Flip(fishMat, fishMat, FlipMode.X); // Unity Y축 반전 복원

        // 마커 내부 영역 크롭 (inner corner = 마커 크기만큼 안쪽)
        int ms = fishSourceMarkerPx;
        int iw = fw - 2 * ms;
        int ih = fh - 2 * ms;
        if (iw <= 0 || ih <= 0)
        {
            Debug.LogError($"[ArucoDetector] fishSourceMarkerPx({ms})가 너무 커 내부 영역 없음 (fish0: {fw}×{fh})");
            return null;
        }

        float scaleX = (float)targetW / iw;
        float scaleY = (float)targetH / ih;
        Debug.Log($"[ArucoDetector] Fish0 마스크 스케일 — " +
                  $"내부영역: {iw}×{ih}px → 출력: {targetW}×{targetH}px " +
                  $"(scaleX={scaleX:F3}, scaleY={scaleY:F3})");

        using Mat fishInner = new Mat(fishMat, new Rect(ms, ms, iw, ih));

        // 마커 크기 기반 스케일 적용 → 워프 출력 크기로 리사이즈
        using Mat fishResized = new Mat();
        Cv2.Resize(fishInner, fishResized, new OpenCvSharp.Size(targetW, targetH));

        // 실루엣 추출: alpha 채널 우선 사용 (투명 배경 PNG), 없으면 RGB 임계값
        // 주의: 흰 배경 PNG는 Unity에서 alpha=255 전체 → maxA>10 이지만 실제 투명도 없음
        //       minA < 245 확인으로 진짜 투명 배경 여부 판별
        using Mat alphaChannel = new Mat();
        Cv2.ExtractChannel(fishResized, alphaChannel, 3);
        Cv2.MinMaxLoc(alphaChannel, out double minA, out double maxA);
        bool hasTransparentBg = (maxA - minA) > 50; // 의미 있는 alpha 변동 = 투명 배경 존재

        using Mat binary = new Mat();
        if (hasTransparentBg)
        {
            Cv2.Threshold(alphaChannel, binary, 127, 255, ThresholdTypes.Binary);
            Debug.Log($"[ArucoDetector] Fish0 마스크: alpha 채널 사용 (min={minA:F0}, max={maxA:F0})");
        }
        else // alpha=255 전체(흰 배경 PNG) → RGB 기반 임계값
        {
            using Mat gray = new Mat();
            Cv2.CvtColor(fishResized, gray, ColorConversionCodes.RGBA2GRAY);
            Cv2.Threshold(gray, binary, backgroundThreshold, 255, ThresholdTypes.BinaryInv);
            Debug.Log($"[ArucoDetector] Fish0 마스크: RGB 임계값 사용 (alpha min={minA:F0}, max={maxA:F0}, 투명배경 없음)");
        }

        using Mat kernel = Cv2.GetStructuringElement(
            MorphShapes.Ellipse, new OpenCvSharp.Size(7, 7));
        Cv2.MorphologyEx(binary, binary, MorphTypes.Close, kernel);

        Cv2.FindContours(binary, out Point[][] contours, out _,
            RetrievalModes.External, ContourApproximationModes.ApproxSimple);

        Mat mask = new Mat(targetH, targetW, MatType.CV_8UC1, new Scalar(0));
        double minArea = targetW * targetH * 0.001;
        for (int i = 0; i < contours.Length; i++)
            if (Cv2.ContourArea(contours[i]) >= minArea)
                Cv2.DrawContours(mask, contours, i, new Scalar(255), thickness: -1);

        return mask; // _cachedFishMask에 저장, OnDestroy에서 Dispose
    }

    // ── 마커 내부 영역 크기 디버그 출력 ──────────────────────
    // 4개 내부 꼭짓점(TL/TR/BR/BL) 사이의 가로·세로 픽셀 크기를 로그로 출력
    // 사용법: Inspector 버튼 또는 코드에서 LogInnerAreaSize(corners, ids) 호출
    public void LogInnerAreaSize(Point2f[][] corners, int[] ids)
    {
        Point2f[] inner = GetInnerCorners(corners, ids);
        // inner[0]=TL, inner[1]=TR, inner[2]=BR, inner[3]=BL
        Point2f tl = inner[0], tr = inner[1], br = inner[2], bl = inner[3];

        float topWidth    = Mathf.Sqrt((tr.X - tl.X) * (tr.X - tl.X) + (tr.Y - tl.Y) * (tr.Y - tl.Y));
        float bottomWidth = Mathf.Sqrt((br.X - bl.X) * (br.X - bl.X) + (br.Y - bl.Y) * (br.Y - bl.Y));
        float leftHeight  = Mathf.Sqrt((bl.X - tl.X) * (bl.X - tl.X) + (bl.Y - tl.Y) * (bl.Y - tl.Y));
        float rightHeight = Mathf.Sqrt((br.X - tr.X) * (br.X - tr.X) + (br.Y - tr.Y) * (br.Y - tr.Y));

        float avgWidth  = (topWidth  + bottomWidth) / 2f;
        float avgHeight = (leftHeight + rightHeight) / 2f;

        Debug.Log($"[ArucoDetector] 마커 내부 영역 크기\n" +
                  $"  Width  → 상단: {topWidth:F1}px / 하단: {bottomWidth:F1}px / 평균: {avgWidth:F1}px\n" +
                  $"  Height → 좌측: {leftHeight:F1}px / 우측: {rightHeight:F1}px / 평균: {avgHeight:F1}px\n" +
                  $"  TL({tl.X:F0},{tl.Y:F0})  TR({tr.X:F0},{tr.Y:F0})\n" +
                  $"  BL({bl.X:F0},{bl.Y:F0})  BR({br.X:F0},{br.Y:F0})");
    }

    // ── 디버그 오버레이 ───────────────────────────────────────
    // Inspector에서 overlayView 지정 시에만 동작
    void DrawOverlay(Mat frame, Point2f[][] corners, int[] ids)
    {
        if (overlayView == null) return;

        using Mat bgr = new Mat();
        Cv2.CvtColor(frame, bgr, ColorConversionCodes.RGBA2BGR);
        using Mat overlay = bgr.Clone();
        CvAruco.DrawDetectedMarkers(overlay, corners, ids);

        using Mat overlayRGBA = new Mat();
        Cv2.CvtColor(overlay, overlayRGBA, ColorConversionCodes.BGR2RGBA);

        // 이전 프레임 텍스처 해제 후 교체 (매 프레임 메모리 누수 방지)
        if (overlayView.texture != null)
            Destroy(overlayView.texture);
        overlayView.texture = MatToTexture2D(overlayRGBA);
    }
}

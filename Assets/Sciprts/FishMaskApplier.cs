using UnityEngine;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using OpenCvSharp;

public class FishMaskApplier : MonoBehaviour
{
    // ── 공개 프로퍼티 ──────────────────────────────────────────
    public int FishCount => _fishCount;

    // ── 내부 상태 ──────────────────────────────────────────────
    private int   _fishCount = 0;
    private Mat[] _alphaMats;   // fish 별 원본 alpha(1ch) Mat (실루엣 기준, 마커 없음)

    // ── 초기화: Resources/FishMaskApplier/fish0, fish1, ... 순서로 로드 ──
    void Start()
    {
        var list = new List<Mat>();
        for (int i = 0; ; i++)
        {
            var tex = Resources.Load<Texture2D>($"FishMaskApplier/fish{i}");
            if (tex == null) break;
            list.Add(TextureToAlphaMat(tex));
            Debug.Log($"[FishMaskApplier] fish{i} 로드 ({tex.width}×{tex.height}px)");
        }

        _fishCount = list.Count;
        _alphaMats = list.ToArray();

        if (_fishCount == 0)
            Debug.LogWarning("[FishMaskApplier] Resources/FishMaskApplier 에 fish*.png 파일이 없습니다.");
        else
            Debug.Log($"[FishMaskApplier] 총 {_fishCount}개 로드 완료 → ID 범위 0~{_fishCount * 4 - 1}");
    }

    void OnDestroy()
    {
        if (_alphaMats == null) return;
        foreach (var m in _alphaMats)
            m?.Dispose();
    }

    // ── 마스크 적용 ───────────────────────────────────────────
    public void Apply(Mat rgba, int fishIndex, Point2f[] innerCorners)
    {
        if (fishIndex < 0 || fishIndex >= _fishCount) return;

        Mat alphaSrc = _alphaMats[fishIndex];
        if (alphaSrc == null || alphaSrc.Empty()) return;

        // ── 내부 영역 bounding box 계산 ───────────────────────
        int left   = Mathf.Max(0, Mathf.RoundToInt(Mathf.Min(innerCorners[0].X, innerCorners[3].X)));
        int top    = Mathf.Max(0, Mathf.RoundToInt(Mathf.Min(innerCorners[0].Y, innerCorners[1].Y)));
        int right  = Mathf.Min(rgba.Cols, Mathf.RoundToInt(Mathf.Max(innerCorners[1].X, innerCorners[2].X)));
        int bottom = Mathf.Min(rgba.Rows, Mathf.RoundToInt(Mathf.Max(innerCorners[2].Y, innerCorners[3].Y)));

        int innerW = Mathf.Max(1, right  - left);
        int innerH = Mathf.Max(1, bottom - top);

        // ── Fish PNG를 내부 영역 크기로 직접 리사이즈 ─────────
        // FishMaskApplier/fish{i}.png 는 마커 없이 물고기 실루엣만 포함한 파일
        using Mat fishResized = new Mat();
        Cv2.Resize(alphaSrc, fishResized,
                   new OpenCvSharp.Size(innerW, innerH),
                   interpolation: InterpolationFlags.Linear);

        // ── 전체 출력 크기의 빈 마스크(alpha=0) 생성 후 내부 영역에 배치 ──
        using Mat fullMask = Mat.Zeros(rgba.Rows, rgba.Cols, MatType.CV_8UC1);
        using Mat roi = new Mat(fullMask,
                   new OpenCvSharp.Rect(left, top, innerW, innerH));
        fishResized.CopyTo(roi);

        // ── alpha 채널 교체 ───────────────────────────────────
        Cv2.MixChannels(new[] { fullMask }, new[] { rgba }, new[] { 0, 3 });

        Debug.Log($"[FishMaskApplier] fish{fishIndex} 마스크 적용 완료 | " +
                  $"내부영역=({left},{top})~({right},{bottom}) {innerW}×{innerH}px | " +
                  $"fishPNG={alphaSrc.Cols}×{alphaSrc.Rows} → 리사이즈={innerW}×{innerH}");
    }

    // ── Texture2D → alpha 1채널 Mat ───────────────────────────
    static Mat TextureToAlphaMat(Texture2D tex)
    {
        Color32[] pixels = tex.GetPixels32();
        byte[] alpha = new byte[pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
            alpha[i] = pixels[i].a;

        Mat mat = new Mat(tex.height, tex.width, MatType.CV_8UC1);
        Marshal.Copy(alpha, 0, mat.Data, alpha.Length);
        Cv2.Flip(mat, mat, FlipMode.X); // Unity Y축 반전 복원
        return mat;
    }
}

using UnityEngine;
using System.Runtime.InteropServices;
using OpenCvSharp;

public class BackgroundRemover : MonoBehaviour
{
    [Range(1, 10)] public int iterations = 5;
    [Range(5, 30)] public int marginPercent = 10;

    public Texture2D RemoveBackground(Texture2D source)
    {
        // ── Texture2D → Mat ──────────────────────────────────
        byte[] rawIn = source.GetRawTextureData();
        Mat src = new Mat(source.height, source.width, MatType.CV_8UC4);
        Marshal.Copy(rawIn, 0, src.Data, rawIn.Length);

        // RGBA → BGR (GrabCut은 BGR 필요)
        Mat bgr = new Mat();
        Cv2.CvtColor(src, bgr, ColorConversionCodes.RGBA2BGR);

        // ── GrabCut 설정 ─────────────────────────────────────
        int m = (int)(source.width * marginPercent / 100f);
        OpenCvSharp.Rect roi = new OpenCvSharp.Rect(
            m, m,
            source.width - m * 2,
            source.height - m * 2);

        // 초기 마스크: 전체 배경(BGD)
        Mat mask = new Mat(
            bgr.Rows, bgr.Cols,
            MatType.CV_8UC1,
            new Scalar((int)GrabCutClasses.BGD));
        Mat bgdMdl = new Mat();
        Mat fgdMdl = new Mat();

        // ROI 내부 → 아마도 포그라운드(PR_FGD)
        using (Mat roiRegion = mask.SubMat(roi))
            roiRegion.SetTo(new Scalar((int)GrabCutClasses.PR_FGD));

        // ── GrabCut 실행 ─────────────────────────────────────
        Cv2.GrabCut(
            bgr, mask, roi,
            bgdMdl, fgdMdl,
            iterations,
            GrabCutModes.InitWithRect);

        // ── 포그라운드 마스크 추출 ────────────────────────────
        Mat fg1 = new Mat();
        Mat fg2 = new Mat();
        Mat fgMask = new Mat();

        Cv2.Compare(mask, new Scalar((int)GrabCutClasses.PR_FGD), fg1, CmpTypes.EQ);
        Cv2.Compare(mask, new Scalar((int)GrabCutClasses.FGD), fg2, CmpTypes.EQ);
        Cv2.BitwiseOr(fg1, fg2, fgMask);

        // ── 알파채널 합성 ─────────────────────────────────────
        Mat rgba = new Mat();
        Cv2.CvtColor(bgr, rgba, ColorConversionCodes.BGR2RGBA);

        Cv2.Split(rgba, out Mat[] channels);
        fgMask.CopyTo(channels[3]);     // 알파 = 포그라운드 마스크
        Cv2.Merge(channels, rgba);

        // ── Mat → Texture2D ──────────────────────────────────
        byte[] rawOut = new byte[source.width * source.height * 4];
        Marshal.Copy(rgba.Data, rawOut, 0, rawOut.Length);

        Texture2D result = new Texture2D(
            source.width, source.height,
            TextureFormat.RGBA32, false);
        result.LoadRawTextureData(rawOut);
        result.Apply();

        // ── 메모리 해제 ───────────────────────────────────────
        src.Dispose(); bgr.Dispose();
        mask.Dispose(); bgdMdl.Dispose(); fgdMdl.Dispose();
        fg1.Dispose(); fg2.Dispose(); fgMask.Dispose();
        rgba.Dispose();
        foreach (var ch in channels) ch.Dispose();

        return result;
    }
}

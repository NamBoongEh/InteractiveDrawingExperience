using UnityEngine;
using System.Runtime.InteropServices;
using OpenCvSharp;

public class BackgroundRemover : MonoBehaviour
{
    /// <summary>
    /// GrabCut 알고리즘 반복 횟수.
    /// 값이 클수록 전경/배경 경계를 더 정밀하게 다듬지만 처리 시간이 늘어난다.
    /// - 낮을수록 (1~3): 빠르지만 경계가 거칠고 부정확할 수 있음
    /// - 높을수록 (8~15): 느리지만 세밀한 경계 추출 가능
    /// - 권장값: 5 (속도와 정확도 균형)
    /// </summary>
    public int iterations = 5;

    /// <summary>
    /// 이미지 가장자리로부터 잘라낼 여백 비율(%).
    /// GrabCut의 ROI(관심 영역)를 이미지 전체 크기에서 얼마나 안쪽으로 설정할지 결정.
    /// - 낮을수록 (0~5): ROI가 가장자리까지 넓어져 물체가 끝까지 잘릴 위험이 줄지만,
    ///   배경 영역이 전경으로 잘못 분류될 수 있음
    /// - 높을수록 (15~30): ROI가 좁아져 이미지 중앙 물체만 전경으로 인식,
    ///   물체가 모서리 근처에 있으면 잘릴 수 있음
    /// - 권장값: 10 (A4 용지 기준, 마커 영역 제외 후 적용 시)
    /// </summary>
    public int marginPercent = 10;

    public Texture2D RemoveBackground(Texture2D source)
    {
        // ���� Texture2D �� Mat ��������������������������������������������������������������������
        byte[] rawIn = source.GetRawTextureData();
        Mat src = new Mat(source.height, source.width, MatType.CV_8UC4);
        Marshal.Copy(rawIn, 0, src.Data, rawIn.Length);

        // RGBA �� BGR (GrabCut�� BGR �ʿ�)
        Mat bgr = new Mat();
        Cv2.CvtColor(src, bgr, ColorConversionCodes.RGBA2BGR);

        // ���� GrabCut ���� ��������������������������������������������������������������������������
        int m = (int)(source.width * marginPercent / 100f);
        OpenCvSharp.Rect roi = new OpenCvSharp.Rect(
            m, m,
            source.width - m * 2,
            source.height - m * 2);

        // �ʱ� ����ũ: ��ü ���(BGD)
        Mat mask = new Mat(
            bgr.Rows, bgr.Cols,
            MatType.CV_8UC1,
            new Scalar((int)GrabCutClasses.BGD));
        Mat bgdMdl = new Mat();
        Mat fgdMdl = new Mat();

        // ROI ���� �� �Ƹ��� ���׶���(PR_FGD)
        using (Mat roiRegion = mask.SubMat(roi))
            roiRegion.SetTo(new Scalar((int)GrabCutClasses.PR_FGD));

        // ���� GrabCut ���� ��������������������������������������������������������������������������
        Cv2.GrabCut(
            bgr, mask, roi,
            bgdMdl, fgdMdl,
            iterations,
            GrabCutModes.InitWithRect);

        // ���� ���׶��� ����ũ ���� ��������������������������������������������������������
        Mat fg1 = new Mat();
        Mat fg2 = new Mat();
        Mat fgMask = new Mat();

        Cv2.Compare(mask, new Scalar((int)GrabCutClasses.PR_FGD), fg1, CmpTypes.EQ);
        Cv2.Compare(mask, new Scalar((int)GrabCutClasses.FGD), fg2, CmpTypes.EQ);
        Cv2.BitwiseOr(fg1, fg2, fgMask);

        // ���� ����ä�� �ռ� ��������������������������������������������������������������������������
        Mat rgba = new Mat();
        Cv2.CvtColor(bgr, rgba, ColorConversionCodes.BGR2RGBA);

        Cv2.Split(rgba, out Mat[] channels);
        fgMask.CopyTo(channels[3]);     // ���� = ���׶��� ����ũ
        Cv2.Merge(channels, rgba);

        // ���� Mat �� Texture2D ��������������������������������������������������������������������
        byte[] rawOut = new byte[source.width * source.height * 4];
        Marshal.Copy(rgba.Data, rawOut, 0, rawOut.Length);

        Texture2D result = new Texture2D(
            source.width, source.height,
            TextureFormat.RGBA32, false);
        result.LoadRawTextureData(rawOut);
        result.Apply();

        // ���� �޸� ���� ������������������������������������������������������������������������������
        src.Dispose(); bgr.Dispose();
        mask.Dispose(); bgdMdl.Dispose(); fgdMdl.Dispose();
        fg1.Dispose(); fg2.Dispose(); fgMask.Dispose();
        rgba.Dispose();
        foreach (var ch in channels) ch.Dispose();

        return result;
    }
}

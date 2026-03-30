using UnityEngine;
using UnityEditor;
using OpenCvSharp;
using System.IO;

/// <summary>
/// Fish/ 원본 외곽선 이미지 → FishMaskOnlyFish/ 채워진 실루엣 마스크 자동 생성.
/// Menu: Tools > Generate FishMaskOnlyFish
///
/// 변환 원리:
///   1. 원본 외곽선(검정 선, 흰 배경)을 로드
///   2. V채널 임계값으로 검정 외곽선 = 장벽 검출
///   3. 이미지 모서리에서 FloodFill → 외부(흰 배경) 마킹
///   4. 외부가 아닌 영역(외곽선 + 내부) = 흰색 → 실루엣 마스크
///   5. Resources/FishMaskOnlyFish/fish{n}.png 로 저장
/// </summary>
public class FishMaskGenerator : MonoBehaviour
{
    [MenuItem("Tools/Generate FishMaskOnlyFish Masks")]
    static void GenerateMasks()
    {
        string srcFolder  = "Assets/Resources/Fish";
        string dstFolder  = "Assets/Resources/FishMaskOnlyFish";
        string dstAbsPath = Path.Combine(Application.dataPath, "Resources/FishMaskOnlyFish");

        if (!Directory.Exists(dstAbsPath))
            Directory.CreateDirectory(dstAbsPath);

        int count = 0;
        for (int i = 0; i < 20; i++)
        {
            string srcPath = $"{srcFolder}/fish{i}.png";
            Texture2D src = AssetDatabase.LoadAssetAtPath<Texture2D>(srcPath);
            if (src == null) break;

            Texture2D mask = GenerateFilledMask(src);
            if (mask == null) { Debug.LogError($"[FishMaskGen] fish{i} 처리 실패"); continue; }

            string dstPath = Path.Combine(dstAbsPath, $"fish{i}.png");
            File.WriteAllBytes(dstPath, mask.EncodeToPNG());
            Object.DestroyImmediate(mask);

            Debug.Log($"[FishMaskGen] 저장: {dstPath}");
            count++;
        }

        AssetDatabase.Refresh();
        Debug.Log($"[FishMaskGen] 완료: {count}개 마스크 생성 → {dstFolder}/");
        EditorUtility.DisplayDialog("FishMask Generator", $"{count}개 마스크 생성 완료!\n{dstFolder}/", "OK");
    }

    static Texture2D GenerateFilledMask(Texture2D src)
    {
        // Texture2D → RGBA byte 배열 → OpenCV Mat
        // readable 여부와 무관하게 RenderTexture 경유로 픽셀 읽기
        RenderTexture rt = RenderTexture.GetTemporary(src.width, src.height, 0, RenderTextureFormat.ARGB32);
        Graphics.Blit(src, rt);
        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = rt;
        Texture2D readable = new Texture2D(src.width, src.height, TextureFormat.RGBA32, false);
        readable.ReadPixels(new UnityEngine.Rect(0, 0, src.width, src.height), 0, 0);
        readable.Apply();
        RenderTexture.active = prev;
        RenderTexture.ReleaseTemporary(rt);

        Color32[] pixels = readable.GetPixels32();
        Object.DestroyImmediate(readable);

        int W = src.width, H = src.height;
        byte[] rgba = new byte[W * H * 4];
        // Unity는 Y축 뒤집힘 → 아래부터 저장 → OpenCV용으로 뒤집기
        for (int y = 0; y < H; y++)
        {
            int srcRow = H - 1 - y; // Unity Y flip 보정
            for (int x = 0; x < W; x++)
            {
                Color32 c = pixels[srcRow * W + x];
                int idx = (y * W + x) * 4;
                rgba[idx    ] = c.r;
                rgba[idx + 1] = c.g;
                rgba[idx + 2] = c.b;
                rgba[idx + 3] = c.a;
            }
        }

        using Mat mat = new Mat(H, W, MatType.CV_8UC4);
        System.Runtime.InteropServices.Marshal.Copy(rgba, 0, mat.Data, rgba.Length);

        // BGR + HSV → V채널 추출
        using Mat bgr = new Mat();
        Cv2.CvtColor(mat, bgr, ColorConversionCodes.RGBA2BGR);
        using Mat hsv = new Mat();
        Cv2.CvtColor(bgr, hsv, ColorConversionCodes.BGR2HSV);
        using Mat vCh = new Mat();
        Cv2.ExtractChannel(hsv, vCh, 2);

        // V < 180 → 장벽(검정 외곽선)
        // 원본이 깨끗한 흰 배경이므로 threshold 높여도 됨
        using Mat barrier = new Mat();
        Cv2.Threshold(vCh, barrier, 180, 255, ThresholdTypes.BinaryInv);

        // 장벽 Dilate → 외곽선 두께 보강 (틈새 막기)
        int kSz = Mathf.Max(3, W / 200) | 1;
        using var dk = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(kSz, kSz));
        Cv2.Dilate(barrier, barrier, dk, iterations: 2);

        // passable = NOT barrier (흰 배경 + 내부 흰 영역)
        using Mat passable = new Mat();
        Cv2.BitwiseNot(barrier, passable);
        passable.SetTo(Scalar.White); // 장벽 아닌 곳 = 255
        // 장벽 위치는 0으로
        using Mat barrierMask = new Mat();
        Cv2.Compare(barrier, new Scalar(128), barrierMask, CmpTypes.GE);
        passable.SetTo(Scalar.Black, barrierMask);

        // FloodFill: 4 모서리에서 외부 채우기 (128로 마킹)
        var seeds = new[]
        {
            new Point(0, 0), new Point(W - 1, 0),
            new Point(0, H - 1), new Point(W - 1, H - 1),
            new Point(W / 2, 0), new Point(W / 2, H - 1),
            new Point(0, H / 2), new Point(W - 1, H / 2),
        };
        foreach (var seed in seeds)
        {
            int sy = seed.Y, sx = seed.X;
            if (sy < 0 || sy >= H || sx < 0 || sx >= W) continue;
            if (passable.At<byte>(sy, sx) < 127) continue;
            Cv2.FloodFill(passable, seed, new Scalar(128));
        }

        // alpha: 128(외부)→0, 나머지(내부+장벽)→255
        using Mat alpha = new Mat(H, W, MatType.CV_8UC1, Scalar.White);
        // 외부(128) 영역 → 0
        using Mat extMask = new Mat();
        Cv2.InRange(passable, new Scalar(127), new Scalar(129), extMask);
        alpha.SetTo(Scalar.Black, extMask);

        // Morphological Close: 내부 잔여 구멍 메우기
        int cSz = Mathf.Max(5, W / 80) | 1;
        using var ck = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(cSz, cSz));
        Cv2.MorphologyEx(alpha, alpha, MorphTypes.Close, ck, iterations: 3);

        // 외곽 약간 침식 (검은 테두리 제거)
        int eSz = Mathf.Max(3, W / 150) | 1;
        using var ek = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(eSz, eSz));
        Cv2.Erode(alpha, alpha, ek, iterations: 2);

        // Texture2D 변환
        byte[] outBytes = new byte[W * H];
        System.Runtime.InteropServices.Marshal.Copy(alpha.Data, outBytes, 0, outBytes.Length);

        Texture2D result = new Texture2D(W, H, TextureFormat.RGBA32, false);
        Color32[] outPixels = new Color32[W * H];
        for (int y = 0; y < H; y++)
        {
            int dstRow = H - 1 - y; // OpenCV → Unity Y flip 복원
            for (int x = 0; x < W; x++)
            {
                byte v = outBytes[y * W + x];
                outPixels[dstRow * W + x] = new Color32(v, v, v, 255);
            }
        }
        result.SetPixels32(outPixels);
        result.Apply();
        return result;
    }
}

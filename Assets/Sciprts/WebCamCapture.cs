using UnityEngine;
using UnityEngine.UI;

public class WebCamCapture : MonoBehaviour
{
    [Header("Camera Settings")]
    public int deviceIndex = 0;
    public int reqWidth = 1920;
    public int reqHeight = 1080;
    public int reqFPS = 30;

    [Header("UI")]
    public RawImage previewImage;

    public WebCamTexture CamTexture { get; private set; }

    public void StartCamera()
    {
        var devices = WebCamTexture.devices;
        if (devices.Length == 0) { Debug.LogError("À¥Ä· ¾øÀ½"); return; }

        string name = devices[Mathf.Clamp(deviceIndex, 0, devices.Length - 1)].name;
        CamTexture = new WebCamTexture(name, reqWidth, reqHeight, reqFPS);
        CamTexture.Play();

        if (previewImage != null)
            previewImage.texture = CamTexture;

        Debug.Log($"[Scanner] À¥Ä· ½ÃÀÛ: {name}");
    }

    public Texture2D CaptureFrame()
    {
        var snap = new Texture2D(CamTexture.width, CamTexture.height, TextureFormat.RGBA32, false);
        snap.SetPixels32(CamTexture.GetPixels32());
        snap.Apply();
        return snap;
    }

    void OnDestroy() => CamTexture?.Stop();
}

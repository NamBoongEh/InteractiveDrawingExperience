using UnityEngine;
using UnityEngine.UI;
using System;
using System.IO;
using System.Threading.Tasks;

public enum ScannerState { Idle, Scanning, Processing, Sending, Done }

public class ScannerManager : MonoBehaviour
{
    public static ScannerManager Instance { get; private set; }

    [Header("References")]
    public WebCamCapture webCam;
    public ArucoDetector arucoDetector;
    public UdpTextSender udpTextSender;

    [Header("FTP Settings")]
    public string ftpHost = "192.168.0.32";
    public int ftpPort = 21;
    public string ftpUser = "ftpuser";
    public string ftpPassword = "1234";

    [Header("UI")]
    public Text statusText;
    public Button scanButton;

    public ScannerState State { get; private set; } = ScannerState.Idle;

    void Awake() => Instance = this;

    void Start()
    {
        scanButton.interactable = true;
        webCam.StartCamera();
        arucoDetector.OnAllMarkersDetected += HandleMarkersDetected;
        SetState(ScannerState.Idle);
    }

    void Update()
    {
        // Enter 키(Return 또는 NumpadEnter)로 스캔 버튼과 동일하게 동작
        if ((Input.GetKeyDown(KeyCode.Return) || Input.GetKeyDown(KeyCode.KeypadEnter))
            && scanButton.interactable)
        {
            StartScanning();
        }
    }

    public void StartScanning()
    {
        scanButton.interactable = false;
        if (State != ScannerState.Idle) return;
        SetState(ScannerState.Scanning);
    }

    async void HandleMarkersDetected(Texture2D warped, int tlMarkerId)
    {
        SetState(ScannerState.Processing);

        string timestamp = DateTime.Now.ToString("yyyyMMddHHmmss");
        int fishNumber = tlMarkerId / 4;
        string fileName = $"{timestamp}_fish{fishNumber}.png";
        string dirPath  = $"D:\\ftp\\fish{fishNumber}";
        string fullPath = Path.Combine(dirPath, fileName);

        // ── 폴더 없으면 1회 생성 ──────────────────────────────
        if (!Directory.Exists(dirPath))
        {
            Directory.CreateDirectory(dirPath);
            Debug.Log($"[저장] 폴더 생성: {dirPath}");
        }

        // ── 직접 저장 (FTP 없이) ─────────────────────────────
        byte[] pngBytes = warped.EncodeToPNG();
        Destroy(warped); // PNG 인코딩 완료 후 즉시 해제 (누수 방지)
        await Task.Run(() => File.WriteAllBytes(fullPath, pngBytes));
        Debug.Log($"[저장] {fullPath}");

        // ── UDP 전송 ─────────────────────────────────────────
        SetState(ScannerState.Sending);
        string fileNameNoExt = Path.GetFileNameWithoutExtension(fileName);
        string message = $"success:{fileNameNoExt}";

        if (udpTextSender == null)
        {
            Debug.LogError("[UDP] udpTextSender가 Inspector에서 할당되지 않음");
            SetState(ScannerState.Idle);
            return;
        }
        bool sent = await udpTextSender.SendAsync(message);

        Debug.Log($"[UDP] 전송: {message}");
        SetState(sent ? ScannerState.Done : ScannerState.Idle);

        if (sent)
        {
            await Task.Delay(2000);
            scanButton.interactable = true;
            SetState(ScannerState.Idle);
        }
    }



    void SetState(ScannerState next)
    {
        State = next;
        statusText.text = next switch
        {
            ScannerState.Idle => "종이를 웹캠 앞에 놓고 [스캔 시작]을 누르세요",
            ScannerState.Scanning => "마커를 감지 중...",
            ScannerState.Processing => "이미지 처리 중...",
            ScannerState.Sending => "FTP 업로드 & 전송 중...",
            ScannerState.Done => "완료!",
            _ => ""
        };
    }
}

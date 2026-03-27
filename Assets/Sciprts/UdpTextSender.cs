using UnityEngine;
using UnityEngine.Events;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

public class UdpTextSender : MonoBehaviour
{
    [Header("UDP (송수신 공용)")]
    public string wallIP = "192.168.1.100";
    [Tooltip("송신·수신 모두 이 포트를 사용합니다.")]
    public int port = 9000;

    [Tooltip("수신된 메시지를 처리할 이벤트.")]
    public UnityEvent<string> onMessageReceived;

    // ── 공용 클라이언트 ────────────────────────────────────────
    private UdpClient  _udpClient;
    private IPEndPoint _wallEndPoint;

    // ── 수신 스레드 ────────────────────────────────────────────
    private Thread               _recvThread;
    private CancellationTokenSource _cts;

    // ── 초기화 ────────────────────────────────────────────────
    void Start()
    {
        _wallEndPoint = new IPEndPoint(IPAddress.Parse(wallIP), port);

        try
        {
            // 포트에 바인딩 → 수신 가능 / 동일 소켓으로 송신도 가능
            _udpClient = new UdpClient(port);
            _cts       = new CancellationTokenSource();
            _recvThread = new Thread(ReceiveLoop) { IsBackground = true };
            _recvThread.Start();
            Debug.Log($"[UDP] 포트 {port}  송수신 시작  →  {wallIP}:{port}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[UDP] 초기화 실패 (port={port}): {e.Message}");
        }
    }

    // ── 수신 루프 (백그라운드 스레드) ─────────────────────────
    void ReceiveLoop()
    {
        IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);
        while (!_cts.IsCancellationRequested)
        {
            try
            {
                byte[] data = _udpClient.Receive(ref remote);
                string msg  = Encoding.UTF8.GetString(data);
                Debug.Log($"[UDP] 수신: {msg}");
                onMessageReceived?.Invoke(msg);
            }
            catch (SocketException)
            {
                // 소켓 닫힘 (OnDestroy) → 정상 종료
                break;
            }
            catch (Exception e)
            {
                Debug.LogError($"[UDP] 수신 오류: {e.Message}");
            }
        }
    }

    // ── 송신 ──────────────────────────────────────────────────
    public async Task<bool> SendAsync(string message)
    {
        try
        {
            byte[] data = Encoding.UTF8.GetBytes(message);
            await _udpClient.SendAsync(data, data.Length, _wallEndPoint);
            Debug.Log($"[UDP] 송신 완료: {message}");
            return true;
        }
        catch (Exception e)
        {
            Debug.LogError($"[UDP] 송신 실패: {e.Message}");
            return false;
        }
    }

    // ── 정리 ──────────────────────────────────────────────────
    void OnDestroy()
    {
        _cts?.Cancel();
        _udpClient?.Close();               // Close()가 Receive() 블로킹 해제
        _recvThread?.Join(500);            // 최대 500ms 스레드 종료 대기
    }
}

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
    [Header("송신 (Send)")]
    public string wallIP   = "192.168.1.100";
    public int    wallPort = 9000;

    [Header("수신 (Receive)")]
    [Tooltip("이 PC에서 수신 대기할 포트. 0 = 수신 비활성화.")]
    public int  listenPort   = 9001;
    [Tooltip("수신된 메시지를 메인 스레드에서 처리할 이벤트.")]
    public UnityEvent<string> onMessageReceived;

    // ── 송신 ──────────────────────────────────────────────────
    private UdpClient  _sendClient;
    private IPEndPoint _wallEndPoint;

    // ── 수신 ──────────────────────────────────────────────────
    private UdpClient         _recvClient;
    private Thread            _recvThread;
    private CancellationTokenSource _cts;

    // 수신 메시지를 메인 스레드로 전달하는 큐
    private readonly System.Collections.Generic.Queue<string> _recvQueue
        = new System.Collections.Generic.Queue<string>();
    private readonly object _queueLock = new object();

    // ── 초기화 ────────────────────────────────────────────────
    void Start()
    {
        // 송신 클라이언트
        _sendClient  = new UdpClient();
        _wallEndPoint = new IPEndPoint(IPAddress.Parse(wallIP), wallPort);

        // 수신 클라이언트
        if (listenPort > 0)
        {
            try
            {
                _recvClient = new UdpClient(listenPort);
                _cts        = new CancellationTokenSource();
                _recvThread = new Thread(ReceiveLoop) { IsBackground = true };
                _recvThread.Start();
                Debug.Log($"[UDP] 수신 대기 시작: 포트 {listenPort}");
            }
            catch (Exception e)
            {
                Debug.LogError($"[UDP] 수신 클라이언트 생성 실패 (port={listenPort}): {e.Message}");
            }
        }
    }

    // ── 매 프레임: 수신 큐 → 메인 스레드 이벤트 발행 ─────────
    void Update()
    {
        while (true)
        {
            string msg;
            lock (_queueLock)
            {
                if (_recvQueue.Count == 0) break;
                msg = _recvQueue.Dequeue();
            }
            Debug.Log($"[UDP] 수신: {msg}");
            onMessageReceived?.Invoke(msg);
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
                byte[] data = _recvClient.Receive(ref remote);
                string msg  = Encoding.UTF8.GetString(data);
                lock (_queueLock)
                    _recvQueue.Enqueue(msg);
            }
            catch (SocketException)
            {
                // 소켓 닫힘 (OnDestroy 호출) → 정상 종료
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
            await _sendClient.SendAsync(data, data.Length, _wallEndPoint);
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
        _recvClient?.Close();  // Receive() 블로킹 해제
        _sendClient?.Close();
    }
}

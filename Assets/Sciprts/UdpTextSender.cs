using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;

public class UdpTextSender : MonoBehaviour
{
    [Header("Wall PC 연결")]
    public string wallIP = "192.168.1.100";
    public int wallPort = 9000;

    private UdpClient udpClient;
    private IPEndPoint wallEndPoint;

    void Start()
    {
        udpClient = new UdpClient();
        wallEndPoint = new IPEndPoint(IPAddress.Parse(wallIP), wallPort);
    }

    public async Task<bool> SendAsync(string message)
    {
        try
        {
            byte[] data = Encoding.UTF8.GetBytes(message);
            await udpClient.SendAsync(data, data.Length, wallEndPoint);

            Debug.Log($"[UDP] 전송 완료: {message}");
            return true;
        }
        catch (Exception e)
        {
            Debug.LogError($"[UDP] 전송 실패: {e.Message}");
            return false;
        }
    }

    void OnDestroy() => udpClient?.Close();
}

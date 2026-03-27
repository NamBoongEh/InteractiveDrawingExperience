using UnityEngine;
using System.Runtime.InteropServices;
using System.Diagnostics;

/// <summary>
/// ScanButton(또는 지정한 RectTransform)에 마우스 커서를 고정합니다.
/// isLocked = true 이면 매 LateUpdate마다 커서를 버튼 중앙으로 강제 이동합니다.
/// </summary>
public class CursorLocker : MonoBehaviour
{
    [Header("설정")]
    [Tooltip("커서를 고정할 UI 요소 (ScanButton의 RectTransform)")]
    public RectTransform target;

    [Tooltip("target이 속한 Canvas")]
    public Canvas canvas;

    [Tooltip("true = 커서 고정 활성화")]
    public bool isLocked = true;

#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
    // ── Windows API ──────────────────────────────────────────
    [DllImport("user32.dll")]
    static extern bool SetCursorPos(int x, int y);

    [DllImport("user32.dll")]
    static extern bool ClientToScreen(System.IntPtr hWnd, ref WinPoint lpPoint);

    [StructLayout(LayoutKind.Sequential)]
    struct WinPoint { public int X; public int Y; }

    System.IntPtr _hwnd;

    void Start()
    {
        // 현재 프로세스의 메인 윈도우 핸들 캐시
        _hwnd = Process.GetCurrentProcess().MainWindowHandle;
    }
#endif

    void LateUpdate()
    {
        if (!isLocked || target == null || canvas == null) return;

        // ① RectTransform 중심을 Unity 스크린 좌표로 변환
        //    (Screen-Space Overlay: camera 불필요, Camera/World: worldCamera 사용)
        Camera cam = canvas.renderMode == RenderMode.ScreenSpaceOverlay
                   ? null
                   : canvas.worldCamera;

        Vector2 screenPos = RectTransformUtility.WorldToScreenPoint(cam, target.position);

#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        // ② Unity 스크린(좌하단 원점) → 윈도우 클라이언트(좌상단 원점) 변환
        var p = new WinPoint
        {
            X = (int)screenPos.x,
            Y = Screen.height - (int)screenPos.y
        };

        // ③ 클라이언트 좌표 → 절대 스크린 좌표 (창모드 위치 보정)
        ClientToScreen(_hwnd, ref p);
        SetCursorPos(p.X, p.Y);
#else
        // Windows 외 플랫폼: 지원 안 됨
        UnityEngine.Debug.LogWarning("[CursorLocker] Windows 전용 기능입니다.");
        isLocked = false;
#endif
    }
}

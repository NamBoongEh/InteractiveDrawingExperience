public static class GameLog
{
    [System.Diagnostics.Conditional("UNITY_EDITOR")]
    public static void Log(string msg) => UnityEngine.Debug.Log(msg);
}

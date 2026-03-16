using SatorImaging.AppWindowUtility;
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

public class ManagerScripts : MonoBehaviour
{
    [Serializable]
    public class Data
    {
        public string wallIP;
        public int wallPort;
        public string ftpHost;
        public int ftpPort;
        public string ftpUser;
        public string ftpPassword;
        public float ResetTime;
        public int backgroundThreshold;
    }

    Dictionary<KeyCode, Action> keyDictionary;

    //flase 0 , true 1
    readonly string[] playerPrefabName = { "Cursor", "Log", "ScreenTop" };
    int[] playerPrefabInt = { 1, 0, 1 };

    //키보드 업데이트문 작동 속도 조절
    float currentTime = 0f;
    float nextTime = 0.3f;

    //리셋
    [SerializeField] bool ResetSwitch = false;
    [SerializeField] private float ResetTotalTime = 0.0f;
    [SerializeField] float ResetTime = 120.0f;

    //로그
    [SerializeField] GameObject log;
    public static Text logText;
    public static int textline;

    [Header("Config Json")]
    [SerializeField] UdpTextSender udpTextSender;
    [SerializeField] ScannerManager scannerManager;
    [SerializeField] ArucoDetector arucoDetector;


    private void Awake()
    {
        Screen.sleepTimeout = SleepTimeout.NeverSleep;
        //멀티 터치 막기
        Input.multiTouchEnabled = false;
        Cursor.visible = false;
        AppWindowUtility.AlwaysOnTop = true;

        Application.targetFrameRate = 60;

        PlayerPrefabsSettingLoad();
        Load();

        keyDictionary = new Dictionary<KeyCode, Action>
        {
            { KeyCode.F1, KeyboardCursor },
            { KeyCode.F2, KeyboardLog },
            { KeyCode.F3, KeyboardAlwaysOnTop },
            { KeyCode.F12, KeyboardPrefabSave },
            { KeyCode.Escape, KeyboardQuit },
        };
    }

    private void Update()
    {
        if (currentTime > 0f)
            currentTime -= Time.deltaTime;
        else
        {
            if (Input.GetKeyDown(KeyCode.F1))
                KeyboardCursor();
            else if (Input.GetKeyDown(KeyCode.F2))
                KeyboardLog();
            else if (Input.GetKeyDown(KeyCode.F3))
                KeyboardAlwaysOnTop();
            else if (Input.GetKeyDown(KeyCode.F12))
                KeyboardPrefabSave();
            else if (Input.GetKeyDown(KeyCode.Escape))
                KeyboardQuit();
        }

        if (ResetSwitch)
        {
            ResetTotalTime += Time.deltaTime;

            if (ResetTotalTime >= ResetTime)
                ResetAll();
        }

        if (Input.GetMouseButtonDown(0))
            IsResetSwitch(true);
    }

    //120초가 지난 후 리셋
    private void ResetAll()
    {
        //Debug.Log("Reset");
    }

    //true이면 RESET 120초 카운트 시작
    public void IsResetSwitch(bool b)
    {
        ResetSwitch = b;
        ResetTotalTime = 0;

        //Debug.Log("ResetTime reset");
    }

    //마우스 커서 숨기기
    void KeyboardCursor()
    {
        if (Cursor.visible)
        {
            Cursor.visible = false;
            playerPrefabInt[0] = 0;
        }
        else
        {
            Cursor.visible = true;
            playerPrefabInt[0] = 1;
        }

        Debug.Log("Cursor.visible = " + Cursor.visible);
    }

    //로그 보이기
    void KeyboardLog()
    {
        if (log.activeSelf)
        {
            log.SetActive(false);
            playerPrefabInt[1] = 0;
        }
        else
        {
            log.SetActive(true);
            playerPrefabInt[1] = 1;
        }

        Log("log.SetActive = " + log.activeSelf);
    }

    //최상단 설정하기
    void KeyboardAlwaysOnTop()
    {
        if (AppWindowUtility.AlwaysOnTop)
        {
            AppWindowUtility.AlwaysOnTop = false;
            playerPrefabInt[2] = 0;
        }
        else
        {
            AppWindowUtility.AlwaysOnTop = true;
            playerPrefabInt[2] = 1;
        }

        Debug.Log("ScreenTop = " + AppWindowUtility.AlwaysOnTop);
    }

    //플레이어 프리팹에 저장하기
    void KeyboardPrefabSave()
    {
        for (int i = 0; i < playerPrefabName.Length; i++)
            PlayerPrefs.SetInt(playerPrefabName[i], playerPrefabInt[i]);

        Debug.Log("Prefab All Save");
    }

    public void KeyboardQuit()
    {
        Application.Quit();
    }

    //플레이어 프리팹 로드
    void PlayerPrefabsSettingLoad()
    {
        for (int i = 0; i < playerPrefabName.Length; i++)
            playerPrefabInt[i] = PlayerPrefs.GetInt(playerPrefabName[i], playerPrefabInt[i]);

        //커서
        if (playerPrefabInt[0].Equals(0))
            Cursor.visible = false;
        else
            Cursor.visible = true;

        //로그
        if (playerPrefabInt[1].Equals(0))
            log.SetActive(false);
        else
            log.SetActive(true);

        //스크린 탑
        if (playerPrefabInt[2].Equals(0))
            AppWindowUtility.AlwaysOnTop = false;
        else
            AppWindowUtility.AlwaysOnTop = true;
    }


    public static void Log(string msg)
    {
        if (textline >= 20)
        {
            textline = 0;
            logText.text = "";
        }

        Debug.Log($"Log: {msg}");
        logText.text += ($"[{DateTime.Now.ToString("HH:mm")}]  {msg}\n");
        textline++;
    }

    //데이터 불러오기
    public void Load()
    {
        string FromJsonData = File.ReadAllText(Application.streamingAssetsPath + "/SaveFile/save.json");

        Data myData = JsonUtility.FromJson<Data>(FromJsonData);

        udpTextSender.wallIP = myData.wallIP;
        udpTextSender.wallPort = myData.wallPort;

        scannerManager.ftpHost = myData.ftpHost;
        scannerManager.ftpPort = myData.ftpPort;
        scannerManager.ftpUser = myData.ftpUser;
        scannerManager.ftpPassword = myData.ftpPassword;

        ResetTime = myData.ResetTime;
    }
}
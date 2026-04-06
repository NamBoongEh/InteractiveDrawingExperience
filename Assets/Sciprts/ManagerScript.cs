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
        public int port;
        public string ftpHost;
        public int ftpPort;
        public string ftpUser;
        public string ftpPassword;
        public float resetTime;
        public MaskOffsetSet[] maskOffsets;
    }

    Dictionary<KeyCode, Action> keyDictionary;

    //flase 0 , true 1
    readonly string[] playerPrefabName = { "Cursor", "Log", "ScreenTop" };
    int[] playerPrefabInt = { 1, 0, 1 };

    //Ű���� ������Ʈ�� �۵� �ӵ� ����
    float currentTime = 0f;
    float nextTime = 0.3f;

    //�α�
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
        //��Ƽ ��ġ ����
        Input.multiTouchEnabled = false;
        Cursor.visible = false;
        AppWindowUtility.AlwaysOnTop = true;

        Application.targetFrameRate = 60;

        PlayerPrefabsSettingLoad();
        Load();

        logText = log.GetComponent<Text>();
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
    }

    //���콺 Ŀ�� �����
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

        GameLog.Log("Cursor.visible = " + Cursor.visible);
        currentTime = nextTime;
    }

    //�α� ���̱�
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
        currentTime = nextTime;
    }

    //�ֻ�� �����ϱ�
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

        GameLog.Log("ScreenTop = " + AppWindowUtility.AlwaysOnTop);
        currentTime = nextTime;
    }

    //�÷��̾� �����տ� �����ϱ�
    void KeyboardPrefabSave()
    {
        for (int i = 0; i < playerPrefabName.Length; i++)
            PlayerPrefs.SetInt(playerPrefabName[i], playerPrefabInt[i]);

        GameLog.Log("Prefab All Save");
        currentTime = nextTime;
    }

    public void KeyboardQuit()
    {
        Application.Quit();
    }

    //�÷��̾� ������ �ε�
    void PlayerPrefabsSettingLoad()
    {
        for (int i = 0; i < playerPrefabName.Length; i++)
            playerPrefabInt[i] = PlayerPrefs.GetInt(playerPrefabName[i], playerPrefabInt[i]);

        //Ŀ��
        if (playerPrefabInt[0].Equals(0))
            Cursor.visible = false;
        else
            Cursor.visible = true;

        //�α�
        if (playerPrefabInt[1].Equals(0))
            log.SetActive(false);
        else
            log.SetActive(true);

        //��ũ�� ž
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

        GameLog.Log($"Log: {msg}");
        logText.text += ($"[{DateTime.Now.ToString("HH:mm")}]  {msg}\n");
        textline++;
    }

    //������ �ҷ�����
    public void Load()
    {
        string FromJsonData = File.ReadAllText(Application.streamingAssetsPath + "/SaveFile/save.json");

        Data myData = JsonUtility.FromJson<Data>(FromJsonData);

        udpTextSender.wallIP = myData.wallIP;
        udpTextSender.port = myData.port;

        scannerManager.ftpHost = myData.ftpHost;
        scannerManager.ftpPort = myData.ftpPort;
        scannerManager.ftpUser = myData.ftpUser;
        scannerManager.ftpPassword = myData.ftpPassword;

        arucoDetector.maskOffsets = myData.maskOffsets;
    }
}
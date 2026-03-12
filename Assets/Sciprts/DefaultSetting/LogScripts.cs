using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class LogScripts : MonoBehaviour
{
    string y;
    string m;
    string d;

    //일반 통신용
    static FileStream txtMSG;
    static string pathMSG;

    //에러용
    static FileStream txtError;
    static string pathError;

    void Awake()
    {
        Date();
        MakeFileMSG();
        MakeFileError();
    }
    void Date()
    {
        y = DateTime.Now.ToString("yyyy");
        m = DateTime.Now.ToString("MM");
        d = DateTime.Now.ToString("dd");
    }

    void MakeFileMSG()
    {
        pathMSG = $"C:/Log/msg/{y}/{m}";
        DirectoryInfo dir = new DirectoryInfo(pathMSG);

        if(!dir.Exists)
            dir.Create();

        pathMSG += $"/{d}.txt";
    }

    void MakeFileError()
    {
        pathError = $"C:/Log/error/{y}/{m}";
        DirectoryInfo dir = new DirectoryInfo(pathError);

        if(!dir.Exists)
            dir.Create();

        pathError += $"/{d}.txt";
    }
   
    void WriteLog(string msg, string stackTrace, LogType type)
    {
        if (type == LogType.Log)
        {
            txtMSG = new FileStream(pathMSG, FileMode.Append);
            StreamWriter txtWriter = new StreamWriter(txtMSG);
            txtWriter.WriteLine($"[{DateTime.Now.ToString("HH:mm:ss")}] {msg}");
            txtWriter.Flush();
            txtWriter.Close();
        }
        else
        {
            txtMSG = new FileStream(pathError, FileMode.Append);
            StreamWriter txtWriter = new StreamWriter(txtMSG);
            txtWriter.WriteLine($"[{DateTime.Now.ToString("HH:mm:ss")}] [{type}]");
            txtWriter.WriteLine(msg);
            txtWriter.Flush();
            txtWriter.Close();
        }
    }

    private void OnEnable()
    {
        Application.logMessageReceived += WriteLog;
    }
    private void OnDisable()
    {
        Application.logMessageReceived -= WriteLog;
    }
}

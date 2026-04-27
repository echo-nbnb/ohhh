using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace OHHH
{
    public class HandTrackingConnection : MonoBehaviour
    {
        public static HandTrackingConnection Instance { get; private set; }

        [Header("连接设置")]
        public string ip = "127.0.0.1";
        public int port = 8889;
        public float reconnectInterval = 3f;

        [Header("调试")]
        public bool debugMode = true;

        private TcpClient client;
        private NetworkStream stream;
        private Thread receiveThread;
        private bool isConnected = false;
        private bool shouldReconnect = true;

        private readonly object dataQueueLock = new object();
        private readonly Queue<HandTrackingData> dataQueue = new Queue<HandTrackingData>();

        public event Action<HandTrackingData> OnHandTrackingReceived;
        public event Action<string> OnConnectionStatusChanged;

        void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
            }
            else
            {
                Destroy(gameObject);
                return;
            }
        }

        void Update()
        {
            lock (dataQueueLock)
            {
                while (dataQueue.Count > 0)
                {
                    var data = dataQueue.Dequeue();
                    OnHandTrackingReceived?.Invoke(data);
                }
            }
        }

        void Start()
        {
            Log($"[Start] ip={ip} port={port}");
            Connect();
        }

        void OnDestroy()
        {
            Disconnect();
        }

        void OnApplicationQuit()
        {
            Disconnect();
        }

        public void Connect()
        {
            Log($"[Connect] isConnected={isConnected}");
            shouldReconnect = true;
            StopAllCoroutines();
            StartCoroutine(ConnectCoroutine());
        }

        private System.Collections.IEnumerator ConnectCoroutine()
        {
            int attempts = 0;
            while (shouldReconnect && !isConnected)
            {
                attempts++;
                Log($"[ConnectCoroutine] attempt {attempts}");
                bool ok = TryConnect();
                if (ok)
                {
                    Log($"[ConnectCoroutine] 启动接收线程");
                    receiveThread = new Thread(ReceiveLoop) { IsBackground = true };
                    receiveThread.Start();
                }
                else if (shouldReconnect && !isConnected)
                {
                    yield return new WaitForSeconds(reconnectInterval);
                }
            }
        }

        private bool TryConnect()
        {
            try
            {
                client = new TcpClient();
                var ar = client.BeginConnect(ip, port, null, null);
                float t0 = Time.time;
                while (!ar.IsCompleted && Time.time - t0 < 5f) { }
                if (ar.IsCompleted && client.Connected)
                {
                    client.EndConnect(ar);
                    stream = client.GetStream();
                    isConnected = true;
                    Log($"[TryConnect] 连接成功");
                    OnConnectionStatusChanged?.Invoke("connected");
                    return true;
                }
                client.Close();
                Log($"[TryConnect] 超时");
                OnConnectionStatusChanged?.Invoke("reconnecting");
                return false;
            }
            catch (Exception e)
            {
                Log($"[TryConnect] 失败: {e.Message}");
                OnConnectionStatusChanged?.Invoke("error");
                return false;
            }
        }

        public void Disconnect()
        {
            shouldReconnect = false;
            if (receiveThread != null)
            {
                try { receiveThread.Abort(); } catch { }
                receiveThread = null;
            }
            if (stream != null)
            {
                try { stream.Close(); } catch { }
                stream = null;
            }
            if (client != null)
            {
                try { client.Close(); } catch { }
                client = null;
            }
            isConnected = false;
            OnConnectionStatusChanged?.Invoke("disconnected");
        }

        private void ReceiveLoop()
        {
            Log($"[ReceiveLoop] 线程启动");
            byte[] buf = new byte[8192];
            var sb = new StringBuilder();

            while (isConnected && stream != null)
            {
                try
                {
                    if (stream.DataAvailable)
                    {
                        int n = stream.Read(buf, 0, buf.Length);
                        if (n > 0)
                        {
                            sb.Append(Encoding.UTF8.GetString(buf, 0, n));
                            bool found;
                            do
                            {
                                string all = sb.ToString();
                                int idx = all.IndexOf('\n');
                                found = idx >= 0;
                                if (found)
                                {
                                    string line = all.Substring(0, idx);
                                    sb.Remove(0, idx + 1);
                                    if (!string.IsNullOrWhiteSpace(line))
                                    {
                                        ProcessMessage(line);
                                    }
                                }
                            } while (found);
                        }
                    }
                    else
                    {
                        Thread.Sleep(10);
                    }
                }
                catch (ThreadAbortException)
                {
                    break;
                }
                catch (Exception e)
                {
                    Log($"[ReceiveLoop] error: {e.GetType().Name}: {e.Message}");
                    Log($"[ReceiveLoop] Stack: {e.StackTrace}");
                    break;
                }
            }

            isConnected = false;
            UnityMainThreadDispatcher.Enqueue(() =>
            {
                OnConnectionStatusChanged?.Invoke("disconnected");
                if (shouldReconnect) Connect();
            });
        }

        private void ProcessMessage(string msg)
        {
            try
            {
                string display = msg.Length > 100 ? msg.Substring(0, 100) + "..." : msg;
                Log($"[ProcessMessage] raw: {display}");

                var data = JsonUtility.FromJson<HandTrackingData>(msg);
                if (data != null)
                {
                    string contourInfo = data.contour != null ? string.Join(",", data.contour) : "null";
                    Log($"[ProcessMessage] 解析OK contour.Count={data.contour?.Length} contour={contourInfo}");
                    lock (dataQueueLock)
                    {
                        dataQueue.Enqueue(data);
                    }
                }
                else
                {
                    Log($"[ProcessMessage] JsonUtility返回null");
                }
            }
            catch (Exception ex)
            {
                Log($"[ProcessMessage] 异常: {ex.GetType().Name}: {ex.Message}");
                Log($"[ProcessMessage] Stack: {ex.StackTrace}");
            }
        }

        public bool IsConnected => isConnected;

        private void Log(string m)
        {
            if (debugMode) Debug.Log($"[HandTracking] {m}");
        }
    }
}

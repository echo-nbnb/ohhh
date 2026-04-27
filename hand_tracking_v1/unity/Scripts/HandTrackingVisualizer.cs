using System.Collections.Generic;
using UnityEngine;
using OHHH;

public class HandTrackingVisualizer : MonoBehaviour
{
    [Header("画布像素尺寸")]
    public Vector2 canvasSize = new Vector2(1920, 1080);

    [Header("显示")]
    public Color fingertipColor = new Color(0, 1f, 0, 0.8f);
    public Color palmCenterColor = new Color(1f, 1f, 0, 0.8f);
    public Color contourColor = new Color(0, 0.8f, 0, 0.6f);
    public Color boundingBoxColor = new Color(1f, 0.5f, 0, 0.8f);
    public float worldScale = 0.01f;

    [Header("调试")]
    public bool debugMode = true;

    private HandTrackingData currentData;
    private List<GameObject> fingertipDots = new List<GameObject>();
    private GameObject palmCenterDot;
    private LineRenderer contourLine;
    private LineRenderer boundingBoxLine;
    private int frameCount = 0;

    void Start()
    {
        Log($"[Start] Canvas: {canvasSize}");
        if (HandTrackingConnection.Instance == null)
            Log("[Start] HandTrackingConnection.Instance is NULL!");
        else
            HandTrackingConnection.Instance.OnHandTrackingReceived += OnHandTrackingReceived;

        CreatePoolObjects();
    }

    void OnDestroy()
    {
        if (HandTrackingConnection.Instance != null)
            HandTrackingConnection.Instance.OnHandTrackingReceived -= OnHandTrackingReceived;
    }

    private void CreatePoolObjects()
    {
        for (int i = 0; i < 5; i++)
        {
            var dot = MakeDot($"Tip_{i}", 0.2f, fingertipColor);
            fingertipDots.Add(dot);
            dot.SetActive(false);
        }

        palmCenterDot = MakeDot("Palm", 0.35f, palmCenterColor);
        contourLine = MakeLine("Contour", contourColor, 0.06f);
        boundingBoxLine = MakeLine("BoundingBox", boundingBoxColor, 0.05f);
        boundingBoxLine.positionCount = 5;
    }

    private GameObject MakeDot(string name, float size, Color color)
    {
        var go = new GameObject(name);
        go.transform.SetParent(transform);
        var sr = go.AddComponent<SpriteRenderer>();
        var tex = new Texture2D(1, 1);
        tex.SetPixel(0, 0, Color.white);
        tex.Apply();
        sr.sprite = Sprite.Create(tex, new Rect(0, 0, 1, 1), new Vector2(0.5f, 0.5f), 1);
        sr.color = color;
        sr.sortingOrder = 100;
        go.transform.localScale = Vector3.one * size;
        return go;
    }

    private LineRenderer MakeLine(string name, Color color, float width)
    {
        var go = new GameObject(name);
        go.transform.SetParent(transform);
        var lr = go.AddComponent<LineRenderer>();
        lr.startColor = lr.endColor = color;
        lr.startWidth = lr.endWidth = width;
        lr.positionCount = 2;
        lr.useWorldSpace = false;
        lr.material = new Material(Shader.Find("Sprites/Default"));
        lr.sortingOrder = 99;
        return lr;
    }

    private void OnHandTrackingReceived(HandTrackingData data)
    {
        currentData = data;
        frameCount++;
        Log($"[OnHandTrackingReceived] contour={data.contour?.Length} palm={data.palm_center[0]},{data.palm_center[1]}");
        UpdateVisualization();
    }

    private Vector3 PixelToWorld(float px, float py)
    {
        float cx = canvasSize.x / 2f;
        float cy = canvasSize.y / 2f;
        return new Vector3(
            (px - cx) * worldScale,
            (cy - py) * worldScale,
            0
        );
    }

    private void UpdateVisualization()
    {
        if (currentData == null || currentData.contour == null || currentData.contour.Length < 14)
        {
            Log($"[UpdateViz] 数据无效 contour={currentData?.contour?.Length}");
            ClearVisualization();
            return;
        }

        if (debugMode && frameCount % 60 == 0)
        {
            Log($"[UpdateViz] palm=({currentData.palm_center[0]},{currentData.palm_center[1]})");
        }

        // 7个轮廓点
        Vector3[] pts = new Vector3[7];
        for (int i = 0; i < 7; i++)
        {
            float x = currentData.contour[i * 2];
            float y = currentData.contour[i * 2 + 1];
            pts[i] = PixelToWorld(x, y);
        }

        // 手掌中心
        float px = currentData.palm_center[0];
        float py = currentData.palm_center[1];
        palmCenterDot.transform.localPosition = PixelToWorld(px, py);
        palmCenterDot.SetActive(true);

        // 轮廓线
        contourLine.SetPositions(pts);
        contourLine.gameObject.SetActive(true);

        // 5个指尖点
        for (int i = 0; i < 5 && i < fingertipDots.Count; i++)
        {
            fingertipDots[i].transform.localPosition = pts[i + 1];
            fingertipDots[i].SetActive(true);
        }

        // 手部范围框
        if (currentData.bounding_box != null && currentData.bounding_box.Length >= 4)
        {
            float bx0 = currentData.bounding_box[0];
            float by0 = currentData.bounding_box[1];
            float bx1 = currentData.bounding_box[2];
            float by1 = currentData.bounding_box[3];

            Vector3[] boxPts = new Vector3[5];
            boxPts[0] = PixelToWorld(bx0, by0);
            boxPts[1] = PixelToWorld(bx1, by0);
            boxPts[2] = PixelToWorld(bx1, by1);
            boxPts[3] = PixelToWorld(bx0, by1);
            boxPts[4] = boxPts[0];

            boundingBoxLine.SetPositions(boxPts);
            boundingBoxLine.gameObject.SetActive(true);
        }
        else
        {
            boundingBoxLine.gameObject.SetActive(false);
        }
    }

    private void ClearVisualization()
    {
        foreach (var d in fingertipDots) if (d != null) d.SetActive(false);
        if (palmCenterDot != null) palmCenterDot.SetActive(false);
        if (contourLine != null) contourLine.gameObject.SetActive(false);
        if (boundingBoxLine != null) boundingBoxLine.gameObject.SetActive(false);
    }

    private void Log(string m)
    {
        if (debugMode) Debug.Log($"[HandViz] {m}");
    }
}

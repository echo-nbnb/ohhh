using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UnityMainThreadDispatcher : MonoBehaviour
{
    private static readonly Queue<Action> _queue = new Queue<Action>();
    private static readonly object _lock = new object();

    void Update()
    {
        lock (_lock)
        {
            while (_queue.Count > 0)
            {
                var action = _queue.Dequeue();
                try
                {
                    action?.Invoke();
                }
                catch (Exception e)
                {
                    Debug.LogException(e);
                }
            }
        }
    }

    public static void Enqueue(Action action)
    {
        lock (_lock)
        {
            _queue.Enqueue(action);
        }
    }
}

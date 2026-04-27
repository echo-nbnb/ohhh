using System;
using UnityEngine;

namespace OHHH
{
    [Serializable]
    public class HandTrackingData
    {
        public string type;
        public int[] palm_center;   // [x, y]
        public int[] wrist;       // [x, y]
        public int[] contour;     // [x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6] 7个点
        public int[] bounding_box; // [x_min, y_min, x_max, y_max]
    }
}

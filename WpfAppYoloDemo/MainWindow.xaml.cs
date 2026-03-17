using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using Microsoft.Win32;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace WpfAppYoloDemo
{
    public partial class MainWindow : System.Windows.Window
    {
        private VisualHost host;
        private InferenceSession session;
        private List<YoloBoxNormalized> outputList;

        public class YoloBoxNormalized
        {
            public float X { get; set; }        // 左上角 x, 0~1
            public float Y { get; set; }        // 左上角 y, 0~1
            public float W { get; set; }        // 寬度, 0~1
            public float H { get; set; }        // 高度, 0~1
            public float Confidence { get; set; }
            public int ClassId { get; set; }
        }

        public MainWindow()
        {
            InitializeComponent();

            host = new VisualHost();
            Canvas.SetLeft(host, 0);
            Canvas.SetTop(host, 0);
            MyCanvas.Children.Add(host);

            outputList = new List<YoloBoxNormalized>();

            // 初始化 ONNX 推論器
            session = new InferenceSession("./yolov8n.onnx");

            // 綁定 SizeChanged，隨 Image 尺寸改變自動重畫
            image.SizeChanged += (s, e) =>
            {
                if (outputList != null && outputList.Count > 0)
                    DrawYoloBoxes();
            };
        }

        private void BrowserImageClick(object sender, RoutedEventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog
            {
                Title = "選擇圖片",
                Filter = "Image Files|*.png;*.jpg;*.jpeg;*.bmp;*.gif"
            };

            if (dialog.ShowDialog() == true)
            {
                BitmapImage bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.UriSource = new System.Uri(dialog.FileName);
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.EndInit();

                image.Source = bitmap;

                RunYolo(dialog.FileName);

                // 立即呼叫 DrawYoloBoxes，SizeChanged 也會再次更新
                DrawYoloBoxes();
            }
        }

        private void RunYolo(string imgFilePath)
        {
            var img = Cv2.ImRead(imgFilePath);

            int targetSize = 640; // YOLOv8 標準輸入大小
            Mat resized = new Mat();
            Cv2.Resize(img, resized, new OpenCvSharp.Size(targetSize, targetSize));

            // Mat 轉 float array 並正規化
            var input = new float[1 * 3 * targetSize * targetSize];
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < targetSize; y++)
                    for (int x = 0; x < targetSize; x++)
                        input[c * targetSize * targetSize + y * targetSize + x] = resized.At<Vec3b>(y, x)[c] / 255.0f;

            var tensor = new DenseTensor<float>(input, new int[] { 1, 3, targetSize, targetSize });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", tensor) };

            using var results = session.Run(inputs);

            var resultTensor = results.First().AsTensor<float>();
            outputList.Clear();
            outputList = ParseYoloOutputNormalized(resultTensor, targetSize);
        }

        public List<YoloBoxNormalized> ParseYoloOutputNormalized(Tensor<float> resultTensor, int targetSize, float confThreshold = 0.25f)
        {
            int numClasses = 80;
            var boxes = new List<YoloBoxNormalized>();
            int numBoxes = resultTensor.Dimensions[1];

            for (int i = 0; i < numBoxes; i++)
            {
                float x = resultTensor[0, i, 0];
                float y = resultTensor[0, i, 1];
                float w = resultTensor[0, i, 2];
                float h = resultTensor[0, i, 3];
                float conf = resultTensor[0, i, 4];

                // 找到 class probability 最大值
                float maxProb = 0;
                int classId = -1;
                for (int c = 0; c < numClasses; c++)
                {
                    float prob = resultTensor[0, i, 5 + c];
                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        classId = c;
                    }
                }

                float finalConf = conf * maxProb;
                if (finalConf > confThreshold)
                {
                    boxes.Add(new YoloBoxNormalized
                    {
                        X = (x - w / 2) / targetSize,
                        Y = (y - h / 2) / targetSize,
                        W = w / targetSize,
                        H = h / targetSize,
                        Confidence = finalConf,
                        ClassId = classId
                    });
                }
            }

            return boxes;
        }

        private void DrawYoloBoxes()
        {
            if (image.Source == null || outputList == null) return;

            //host.ClearVisuals();

            var bitmap = image.Source as BitmapSource;
            if (bitmap == null) return;

            double imgW = bitmap.PixelWidth;
            double imgH = bitmap.PixelHeight;

            // 使用 RenderSize，避免 ActualWidth=0
            double containerW = image.RenderSize.Width;
            double containerH = image.RenderSize.Height;

            // Uniform 縮放比例
            double scale = System.Math.Min(containerW / imgW, containerH / imgH);

            // 計算留黑邊偏移
            double offsetX = (containerW - imgW * scale) / 2;
            double offsetY = (containerH - imgH * scale) / 2;

            foreach (var box in outputList)
            {
                double x = box.X * imgW * scale + offsetX;
                double y = box.Y * imgH * scale + offsetY;
                double w = box.W * imgW * scale;
                double h = box.H * imgH * scale;

                string text = $"Class:{box.ClassId} Conf:{box.Confidence:F2}";
                host.AddRectangleWithText(new System.Windows.Rect(x, y, w, h), text);
            }
        }
    }
}
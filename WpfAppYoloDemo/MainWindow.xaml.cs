using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp; // OpenCV 核心
using OpenCvSharp.Dnn; // 必須引用這個才能用 NMSBoxes
using OpenCvSharp.WpfExtensions;
using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace WpfAppYoloDemo
{
    public partial class MainWindow : System.Windows.Window
    {
        private VisualHost host;
        private InferenceSession session;
        private List<YoloBox> outputList = new List<YoloBox>();

        private float _ratio = 1.0f;
        private int _padX = 0;
        private int _padY = 0;

        private readonly string[] _classNames = new string[]
{
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "Kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

        public class YoloBox
        {
            public float X { get; set; }
            public float Y { get; set; }
            public float W { get; set; }
            public float H { get; set; }
            public float Confidence { get; set; }
            public int ClassId { get; set; }
        }

        public MainWindow()
        {
            InitializeComponent();
            host = new VisualHost();
            MyCanvas.Children.Add(host);
            session = new InferenceSession("./yolov8n.onnx");

            image.SizeChanged += (s, e) =>
            {
                if (outputList != null && outputList.Count > 0)
                {
                    DrawYoloBoxes();
                }
            };
        }

        private void BrowserImageClick(object sender, RoutedEventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog { Filter = "Image Files|*.png;*.jpg;*.jpeg;*.bmp" };
            if (dialog.ShowDialog() == true)
            {
                BitmapImage bitmap = new BitmapImage(new Uri(dialog.FileName));
                image.Source = bitmap;
                RunYolo(dialog.FileName);
                Dispatcher.BeginInvoke(new Action(() => { DrawYoloBoxes(); }), DispatcherPriority.Render);
            }
        }

        private void RunYolo(string imgFilePath)
        {
            using var img = Cv2.ImRead(imgFilePath);
            int targetSize = 640;

            using var processedImg = PrepareLetterbox(img, new OpenCvSharp.Size(targetSize, targetSize));

            var input = new float[1 * 3 * targetSize * targetSize];
            for (int y = 0; y < targetSize; y++)
            {
                for (int x = 0; x < targetSize; x++)
                {
                    var pixel = processedImg.At<Vec3b>(y, x);
                    input[0 * 640 * 640 + y * 640 + x] = pixel[2] / 255.0f;
                    input[1 * 640 * 640 + y * 640 + x] = pixel[1] / 255.0f;
                    input[2 * 640 * 640 + y * 640 + x] = pixel[0] / 255.0f;
                }
            }

            var tensor = new DenseTensor<float>(input, new int[] { 1, 3, targetSize, targetSize });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", tensor) };

            using var results = session.Run(inputs);
            var resultTensor = results.First().AsTensor<float>();

            outputList = ParseOutput(resultTensor, targetSize);
        }

        private Mat PrepareLetterbox(Mat src, OpenCvSharp.Size targetSize)
        {
            float w = src.Width;
            float h = src.Height;
            _ratio = Math.Min((float)targetSize.Width / w, (float)targetSize.Height / h);

            int newW = (int)Math.Round(w * _ratio);
            int newH = (int)Math.Round(h * _ratio);
            _padX = (targetSize.Width - newW) / 2;
            _padY = (targetSize.Height - newH) / 2;

            Mat resized = new Mat();
            Cv2.Resize(src, resized, new OpenCvSharp.Size(newW, newH));

            Mat dst = new Mat(targetSize, src.Type(), new Scalar(114, 114, 114));

            // 修正：明確指定使用 OpenCvSharp.Rect 避免與 WPF 衝突
            OpenCvSharp.Rect roi = new OpenCvSharp.Rect(_padX, _padY, newW, newH);

            using (Mat mask = new Mat(dst, roi))
            {
                resized.CopyTo(mask);
            }
            return dst;
        }

        private List<YoloBox> ParseOutput(Tensor<float> result, int targetSize)
        {
            int numClasses = 80;
            int numBoxes = result.Dimensions[2];
            var candidateBoxes = new List<Rect2d>();
            var confidences = new List<float>();
            var classIds = new List<int>();

            for (int i = 0; i < numBoxes; i++)
            {
                float maxProb = 0;
                int classId = -1;
                for (int c = 0; c < numClasses; c++)
                {
                    float prob = result[0, 4 + c, i];
                    if (prob > maxProb) { maxProb = prob; classId = c; }
                }

                if (maxProb > 0.45f)
                {
                    float cx = result[0, 0, i];
                    float cy = result[0, 1, i];
                    float w = result[0, 2, i];
                    float h = result[0, 3, i];

                    candidateBoxes.Add(new Rect2d(cx - w / 2, cy - h / 2, w, h));
                    confidences.Add(maxProb);
                    classIds.Add(classId);
                }
            }

            // 修正：Cv2.Dnn 改為 CvDnn.NMSBoxes
            CvDnn.NMSBoxes(candidateBoxes, confidences, 0.45f, 0.5f, out int[] indices);

            var finalResults = new List<YoloBox>();
            float usableW = targetSize - 2 * _padX;
            float usableH = targetSize - 2 * _padY;

            foreach (var idx in indices)
            {
                var rect = candidateBoxes[idx];
                finalResults.Add(new YoloBox
                {
                    X = (float)((rect.X - _padX) / usableW),
                    Y = (float)((rect.Y - _padY) / usableH),
                    W = (float)(rect.Width / usableW),
                    H = (float)(rect.Height / usableH),
                    Confidence = confidences[idx],
                    ClassId = classIds[idx]
                });
            }
            return finalResults;
        }

        private void DrawYoloBoxes()
        {
            if (image.Source == null || outputList == null || !image.IsLoaded) return;

            host.ClearVisuals();
            var bitmap = (BitmapSource)image.Source;

            System.Windows.Point imgPos = image.TranslatePoint(new System.Windows.Point(0, 0), MyCanvas);

            double scale = Math.Min(image.ActualWidth / bitmap.PixelWidth, image.ActualHeight / bitmap.PixelHeight);
            double displayW = bitmap.PixelWidth * scale;
            double displayH = bitmap.PixelHeight * scale;

            double baseX = imgPos.X + (image.ActualWidth - displayW) / 2;
            double baseY = imgPos.Y + (image.ActualHeight - displayH) / 2;

            foreach (var box in outputList)
            {
                double x = baseX + (box.X * displayW);
                double y = baseY + (box.Y * displayH);
                double w = box.W * displayW;
                double h = box.H * displayH;

                // 這裡使用 WPF 的 Rect，不需要 OpenCvSharp 前綴
                string label = (box.ClassId >= 0 && box.ClassId < _classNames.Length)
                   ? _classNames[box.ClassId]
                   : $"ID:{box.ClassId}";

                string text = $"{label} {box.Confidence:P0}"; // 例如：person 95%

                host.AddRectangleWithText(new System.Windows.Rect(x, y, w, h), text);
            }
        }
    }
}
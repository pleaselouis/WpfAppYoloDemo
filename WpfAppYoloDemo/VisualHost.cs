using System.Windows;
using System.Windows.Media;
using System.Collections.Generic;

public class VisualHost : FrameworkElement
{
    private List<DrawingVisual> visuals = new List<DrawingVisual>();

    public VisualHost()
    {
        // 可初始化預設圖形，如果需要
    }

    /// <summary>
    /// 新增矩形與文字
    /// </summary>
    public DrawingVisual AddRectangleWithText(Rect rect, string text)
    {
        DrawingVisual visual = new DrawingVisual();
        using (DrawingContext dc = visual.RenderOpen())
        {
            SolidColorBrush redBrush = new SolidColorBrush(Color.FromArgb(185, 255, 0, 0));
            dc.DrawRectangle(
               null,                        // 填充為 null → 內部透明
               new Pen(redBrush, 2),   
               rect
           );

            // 半透明紅色文字            
            FormattedText formattedText = new FormattedText(
                text,
                System.Globalization.CultureInfo.InvariantCulture,
                FlowDirection.LeftToRight,
                new Typeface("Segoe UI"),
                16,
                redBrush,
                VisualTreeHelper.GetDpi(this).PixelsPerDip
            );

            // 文字置於矩形內
            Point textPos = new Point(rect.X + 10, rect.Y + 10);
            dc.DrawText(formattedText, textPos);
        }

        visuals.Add(visual);
        this.AddVisualChild(visual);
        this.AddLogicalChild(visual);
        return visual; // 回傳以便後續操作
    }

    /// <summary>
    /// 移動已存在的 DrawingVisual
    /// </summary>
    public void MoveVisual(DrawingVisual visual, Vector offset)
    {
        if (!visuals.Contains(visual)) return;

        // 取得原本內容，重畫
        using (DrawingContext dc = visual.RenderOpen())
        {
            // 取得原本矩形與文字（假設你自己存資料或重畫）
            // 這裡示範簡單偏移
            Rect rect = new Rect(offset.X, offset.Y, 100, 60);
            dc.DrawRectangle(Brushes.LightBlue, new Pen(Brushes.Black, 2), rect);

            FormattedText formattedText = new FormattedText(
                "Moved",
                System.Globalization.CultureInfo.InvariantCulture,
                FlowDirection.LeftToRight,
                new Typeface("Segoe UI"),
                16,
                Brushes.Black,
                VisualTreeHelper.GetDpi(this).PixelsPerDip
            );
            dc.DrawText(formattedText, new Point(rect.X + 10, rect.Y + 10));
        }
    }

    /// <summary>
    /// 刪除矩形與文字
    /// </summary>
    public void RemoveVisual(DrawingVisual visual)
    {
        if (visuals.Contains(visual))
        {
            visuals.Remove(visual);
            this.RemoveVisualChild(visual);
            this.RemoveLogicalChild(visual);
        }
    }
    public void ClearVisuals()
    {
        foreach (var visual in visuals)
        {
            this.RemoveVisualChild(visual);
            this.RemoveLogicalChild(visual);
        }
        visuals.Clear();

        // 通知 WPF 重新繪製介面
        this.InvalidateVisual();
    }

    protected override int VisualChildrenCount => visuals.Count;
    protected override Visual GetVisualChild(int index) => visuals[index];
}
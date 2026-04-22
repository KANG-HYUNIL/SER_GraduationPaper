param(
    [string]$OutputDir = "LateX_Paper/undergraduate-thesis/undergraduate-thesis/images/chapter4_v1_models"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

function New-Brush([string]$hex) {
    New-Object System.Drawing.SolidBrush([System.Drawing.ColorTranslator]::FromHtml($hex))
}

function New-Pen([string]$hex, [float]$width = 2.0) {
    $pen = New-Object System.Drawing.Pen([System.Drawing.ColorTranslator]::FromHtml($hex), $width)
    $pen.LineJoin = [System.Drawing.Drawing2D.LineJoin]::Round
    $pen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
    $pen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round
    return $pen
}

function New-FontPx([float]$size, [System.Drawing.FontStyle]$style = [System.Drawing.FontStyle]::Regular) {
    New-Object System.Drawing.Font("Microsoft YaHei", $size, $style, [System.Drawing.GraphicsUnit]::Pixel)
}

function Escape-Xml([string]$text) {
    [System.Security.SecurityElement]::Escape($text)
}

function New-Block {
    param(
        [string]$Id,
        [double]$X,
        [double]$Y,
        [double]$W,
        [double]$H,
        [string]$Text,
        [string]$Fill = "#EEF3F7",
        [string]$Stroke = "#61788A"
    )

    @{
        Id = $Id
        X = [double]$X
        Y = [double]$Y
        W = [double]$W
        H = [double]$H
        Text = $Text
        Fill = $Fill
        Stroke = $Stroke
    }
}

function New-Edge {
    param(
        [string]$From,
        [string]$To,
        [string]$FromSide = "right",
        [string]$ToSide = "left",
        [string]$Route = "horizontal"
    )

    @{
        From = $From
        To = $To
        FromSide = $FromSide
        ToSide = $ToSide
        Route = $Route
    }
}

function Get-AnchorPoint {
    param(
        [hashtable]$Block,
        [string]$Side
    )

    switch ($Side) {
        "left"   { return @{ X = $Block.X; Y = $Block.Y + ($Block.H / 2.0) } }
        "right"  { return @{ X = $Block.X + $Block.W; Y = $Block.Y + ($Block.H / 2.0) } }
        "top"    { return @{ X = $Block.X + ($Block.W / 2.0); Y = $Block.Y } }
        "bottom" { return @{ X = $Block.X + ($Block.W / 2.0); Y = $Block.Y + $Block.H } }
        default  { throw "Unsupported anchor side: $Side" }
    }
}

function Draw-RoundedBox {
    param(
        [System.Drawing.Graphics]$Graphics,
        [hashtable]$Block
    )

    $rect = [System.Drawing.RectangleF]::new([single]$Block.X, [single]$Block.Y, [single]$Block.W, [single]$Block.H)
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $radius = 11.0
    $d = $radius * 2.0
    $path.AddArc($rect.X, $rect.Y, $d, $d, 180, 90)
    $path.AddArc($rect.Right - $d, $rect.Y, $d, $d, 270, 90)
    $path.AddArc($rect.Right - $d, $rect.Bottom - $d, $d, $d, 0, 90)
    $path.AddArc($rect.X, $rect.Bottom - $d, $d, $d, 90, 90)
    $path.CloseFigure()

    $fillBrush = New-Brush $Block.Fill
    $strokePen = New-Pen $Block.Stroke 2.1
    $Graphics.FillPath($fillBrush, $path)
    $Graphics.DrawPath($strokePen, $path)

    $font = New-FontPx 18
    $small = New-FontPx 14
    $textBrush = New-Brush "#1F2B37"
    $fmt = New-Object System.Drawing.StringFormat
    $fmt.Alignment = [System.Drawing.StringAlignment]::Center
    $fmt.LineAlignment = [System.Drawing.StringAlignment]::Center

    $lines = $Block.Text -split "`n"
    if ($lines.Count -le 2) {
        $Graphics.DrawString($Block.Text, $font, $textBrush, $rect, $fmt)
    } else {
        $titleRect = [System.Drawing.RectangleF]::new([single]($rect.X + 8), [single]($rect.Y + 7), [single]($rect.Width - 16), [single]28)
        $bodyRect = [System.Drawing.RectangleF]::new([single]($rect.X + 10), [single]($rect.Y + 36), [single]($rect.Width - 20), [single]($rect.Height - 44))
        $Graphics.DrawString($lines[0], $font, $textBrush, $titleRect, $fmt)
        $Graphics.DrawString(($lines[1..($lines.Count - 1)] -join "`n"), $small, $textBrush, $bodyRect, $fmt)
    }

    $fillBrush.Dispose()
    $strokePen.Dispose()
    $path.Dispose()
    $font.Dispose()
    $small.Dispose()
    $textBrush.Dispose()
    $fmt.Dispose()
}

function Draw-ArrowHead {
    param(
        [System.Drawing.Graphics]$Graphics,
        [double]$X,
        [double]$Y,
        [double]$Angle
    )

    $length = 12.0
    $width = 5.0
    $p1 = New-Object System.Drawing.PointF([float]$X, [float]$Y)
    $p2 = New-Object System.Drawing.PointF(
        [float]($X - $length * [Math]::Cos($Angle) + $width * [Math]::Sin($Angle)),
        [float]($Y - $length * [Math]::Sin($Angle) - $width * [Math]::Cos($Angle))
    )
    $p3 = New-Object System.Drawing.PointF(
        [float]($X - $length * [Math]::Cos($Angle) - $width * [Math]::Sin($Angle)),
        [float]($Y - $length * [Math]::Sin($Angle) + $width * [Math]::Cos($Angle))
    )
    $brush = New-Brush "#4C6274"
    $Graphics.FillPolygon($brush, @($p1, $p2, $p3))
    $brush.Dispose()
}

function Draw-OrthogonalArrow {
    param(
        [System.Drawing.Graphics]$Graphics,
        [hashtable]$Start,
        [hashtable]$End,
        [string]$Route = "horizontal"
    )

    $pen = New-Pen "#4C6274" 2.6

    if ($Route -eq "vertical") {
        $midY = ($Start.Y + $End.Y) / 2.0
        $Graphics.DrawLine($pen, [float]$Start.X, [float]$Start.Y, [float]$Start.X, [float]$midY)
        $Graphics.DrawLine($pen, [float]$Start.X, [float]$midY, [float]$End.X, [float]$midY)
        $Graphics.DrawLine($pen, [float]$End.X, [float]$midY, [float]$End.X, [float]$End.Y)
        if ($End.Y -ge $midY) {
            Draw-ArrowHead -Graphics $Graphics -X $End.X -Y $End.Y -Angle ([Math]::PI / 2.0)
        } else {
            Draw-ArrowHead -Graphics $Graphics -X $End.X -Y $End.Y -Angle (-[Math]::PI / 2.0)
        }
    } else {
        $midX = ($Start.X + $End.X) / 2.0
        $Graphics.DrawLine($pen, [float]$Start.X, [float]$Start.Y, [float]$midX, [float]$Start.Y)
        $Graphics.DrawLine($pen, [float]$midX, [float]$Start.Y, [float]$midX, [float]$End.Y)
        $Graphics.DrawLine($pen, [float]$midX, [float]$End.Y, [float]$End.X, [float]$End.Y)
        if ($End.X -ge $midX) {
            Draw-ArrowHead -Graphics $Graphics -X $End.X -Y $End.Y -Angle 0.0
        } else {
            Draw-ArrowHead -Graphics $Graphics -X $End.X -Y $End.Y -Angle ([Math]::PI)
        }
    }

    $pen.Dispose()
}

function Render-PngDiagram {
    param(
        [hashtable]$Diagram,
        [string]$PngPath
    )

    $bitmap = New-Object System.Drawing.Bitmap($Diagram.Width, $Diagram.Height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $graphics.Clear([System.Drawing.Color]::White)

    foreach ($block in $Diagram.Blocks) {
        Draw-RoundedBox -Graphics $graphics -Block $block
    }

    $blockMap = @{}
    foreach ($block in $Diagram.Blocks) {
        $blockMap[$block.Id] = $block
    }

    foreach ($edge in $Diagram.Edges) {
        $start = Get-AnchorPoint -Block $blockMap[$edge.From] -Side $edge.FromSide
        $end = Get-AnchorPoint -Block $blockMap[$edge.To] -Side $edge.ToSide
        Draw-OrthogonalArrow -Graphics $graphics -Start $start -End $end -Route $edge.Route
    }

    $bitmap.Save($PngPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose()
    $bitmap.Dispose()
}

function Build-DrawioXml {
    param(
        [hashtable]$Diagram
    )

    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine('<mxfile host="app.diagrams.net" modified="2026-04-22T00:00:00.000Z" agent="Codex" version="24.7.17">')
    [void]$sb.AppendLine(("  <diagram name=""{0}"" id=""{1}"">" -f (Escape-Xml $Diagram.Name), ([guid]::NewGuid().ToString("N"))))
    [void]$sb.AppendLine(("    <mxGraphModel dx=""1600"" dy=""900"" grid=""1"" gridSize=""10"" guides=""1"" tooltips=""1"" connect=""1"" arrows=""1"" fold=""1"" page=""1"" pageScale=""1"" pageWidth=""{0}"" pageHeight=""{1}"" math=""0"" shadow=""0"">" -f $Diagram.Width, $Diagram.Height))
    [void]$sb.AppendLine('      <root>')
    [void]$sb.AppendLine('        <mxCell id="0" />')
    [void]$sb.AppendLine('        <mxCell id="1" parent="0" />')

    foreach ($block in $Diagram.Blocks) {
        $style = "rounded=1;whiteSpace=wrap;html=1;fillColor=$($block.Fill);strokeColor=$($block.Stroke);strokeWidth=2;fontSize=17;fontColor=#1F2B37;"
        [void]$sb.AppendLine(("        <mxCell id=""{0}"" value=""{1}"" style=""{2}"" vertex=""1"" parent=""1"">" -f $block.Id, (Escape-Xml $block.Text).Replace("`n", '&#xa;'), $style))
        [void]$sb.AppendLine(("          <mxGeometry x=""{0}"" y=""{1}"" width=""{2}"" height=""{3}"" as=""geometry"" />" -f [int]$block.X, [int]$block.Y, [int]$block.W, [int]$block.H))
        [void]$sb.AppendLine('        </mxCell>')
    }

    $edgeIndex = 0
    foreach ($edge in $Diagram.Edges) {
        $edgeIndex += 1
        $style = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=classic;endFill=1;strokeColor=#4C6274;strokeWidth=2.2;"
        [void]$sb.AppendLine(("        <mxCell id=""edge_{0}"" style=""{1}"" edge=""1"" parent=""1"" source=""{2}"" target=""{3}"">" -f $edgeIndex, $style, $edge.From, $edge.To))
        [void]$sb.AppendLine('          <mxGeometry relative="1" as="geometry" />')
        [void]$sb.AppendLine('        </mxCell>')
    }

    [void]$sb.AppendLine('      </root>')
    [void]$sb.AppendLine('    </mxGraphModel>')
    [void]$sb.AppendLine('  </diagram>')
    [void]$sb.AppendLine('</mxfile>')
    $sb.ToString()
}

if (Test-Path -LiteralPath $OutputDir) {
    Get-ChildItem -LiteralPath $OutputDir -File | Remove-Item -Force
} else {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$diagrams = @(
    @{
        Name = "cnn_baseline_architecture"
        Width = 1380
        Height = 520
        Blocks = @(
            (New-Block "b1" 40 90 180 92 "对数梅尔频谱"),
            (New-Block "b2" 300 60 220 132 "卷积块一`n三乘三卷积`n批归一化 + 激活`n二乘二最大池化"),
            (New-Block "b3" 600 60 220 132 "卷积块二`n三乘三卷积`n批归一化 + 激活`n二乘二最大池化"),
            (New-Block "b4" 900 60 220 132 "卷积块三`n三乘三卷积`n批归一化 + 激活`n二乘二最大池化"),
            (New-Block "b5" 900 300 220 132 "卷积块四`n三乘三卷积`n批归一化 + 激活`n二乘二最大池化"),
            (New-Block "b6" 600 320 180 92 "自适应平均池化`n四乘四"),
            (New-Block "b7" 340 320 160 92 "展平 + 丢弃层"),
            (New-Block "b8" 80 320 180 92 "线性分类器`n八类输出")
        )
        Edges = @(
            (New-Edge "b1" "b2"),
            (New-Edge "b2" "b3"),
            (New-Edge "b3" "b4"),
            (New-Edge "b4" "b5" "bottom" "top" "vertical"),
            (New-Edge "b5" "b6" "left" "right"),
            (New-Edge "b6" "b7" "left" "right"),
            (New-Edge "b7" "b8" "left" "right")
        )
    },
    @{
        Name = "pure_transformer_architecture"
        Width = 1380
        Height = 600
        Blocks = @(
            (New-Block "p0" 470 40 420 420 "编码器堆叠`n五层重复编码器层`n每层包含多头自注意力与前馈网络" "#F8FAFB"),
            (New-Block "p1" 90 460 190 92 "输入频谱图"),
            (New-Block "p2" 340 460 220 92 "二维分块嵌入`n三十二乘三十二"),
            (New-Block "p3" 620 460 180 92 "位置编码"),
            (New-Block "p4" 560 120 240 96 "编码器层"),
            (New-Block "p5" 560 240 240 96 "编码器层"),
            (New-Block "p6" 560 360 240 96 "编码器层"),
            (New-Block "p7" 980 130 200 92 "序列池化"),
            (New-Block "p8" 980 250 200 92 "线性分类器"),
            (New-Block "p9" 980 370 200 92 "八类输出")
        )
        Edges = @(
            (New-Edge "p1" "p2"),
            (New-Edge "p2" "p3"),
            (New-Edge "p3" "p6"),
            (New-Edge "p4" "p5" "bottom" "top" "vertical"),
            (New-Edge "p5" "p6" "bottom" "top" "vertical"),
            (New-Edge "p4" "p7"),
            (New-Edge "p7" "p8" "bottom" "top" "vertical"),
            (New-Edge "p8" "p9" "bottom" "top" "vertical")
        )
    },
    @{
        Name = "transformer_encoder_block"
        Width = 1380
        Height = 520
        Blocks = @(
            (New-Block "t1" 70 90 220 92 "输入标记序列"),
            (New-Block "t2" 420 70 280 126 "多头自注意力`n计算标记之间的全局相关性"),
            (New-Block "t3" 840 70 240 92 "残差连接 + LayerNorm"),
            (New-Block "t4" 840 300 280 116 "前馈网络`n线性层 → 激活函数 → 线性层"),
            (New-Block "t5" 420 300 260 92 "残差连接 + LayerNorm"),
            (New-Block "t6" 70 300 220 92 "输出标记序列")
        )
        Edges = @(
            (New-Edge "t1" "t2"),
            (New-Edge "t2" "t3"),
            (New-Edge "t3" "t4" "bottom" "top" "vertical"),
            (New-Edge "t4" "t5" "left" "right"),
            (New-Edge "t5" "t6" "left" "right")
        )
    },
    @{
        Name = "bridged_window_transformer_architecture"
        Width = 1480
        Height = 760
        Blocks = @(
            (New-Block "w1" 60 90 180 92 "对数梅尔频谱"),
            (New-Block "w2" 340 70 220 126 "卷积前端`n两层卷积压缩`n提取局部时频模式"),
            (New-Block "w3" 660 90 180 92 "空间投影模块"),
            (New-Block "w4" 940 60 240 136 "第一阶段窗口编码`n局部窗口注意力`n移位窗口`n相对位置偏置"),
            (New-Block "w5" 940 280 240 136 "桥接上下文模块一`n聚合阶段级全局摘要`n回注入第一阶段特征"),
            (New-Block "w6" 660 500 180 92 "分块合并"),
            (New-Block "w7" 340 470 220 126 "第二阶段窗口编码`n更大感受野`n更高语义层级"),
            (New-Block "w8" 60 470 220 126 "桥接上下文模块二`n最终全局融合"),
            (New-Block "w9" 60 620 180 92 "均值池化"),
            (New-Block "w10" 340 620 180 92 "八类输出")
        )
        Edges = @(
            (New-Edge "w1" "w2"),
            (New-Edge "w2" "w3"),
            (New-Edge "w3" "w4"),
            (New-Edge "w4" "w5" "bottom" "top" "vertical"),
            (New-Edge "w5" "w6" "bottom" "top" "vertical"),
            (New-Edge "w6" "w7" "left" "right"),
            (New-Edge "w7" "w8" "left" "right"),
            (New-Edge "w8" "w9" "bottom" "top" "vertical"),
            (New-Edge "w9" "w10")
        )
    },
    @{
        Name = "bridge_context_detail"
        Width = 1380
        Height = 520
        Blocks = @(
            (New-Block "g1" 70 80 200 92 "阶段特征图"),
            (New-Block "g2" 380 60 240 126 "展平为空间标记`n保留局部窗口编码后的特征"),
            (New-Block "g3" 760 60 220 126 "可学习桥接标记`n作为全局查询"),
            (New-Block "g4" 560 280 240 126 "交叉注意力聚合`n从全部空间标记中提取摘要"),
            (New-Block "g5" 200 280 220 126 "全局情绪摘要`n形成阶段级上下文"),
            (New-Block "g6" 200 430 240 92 "重加权或投影回注"),
            (New-Block "g7" 560 430 220 92 "更新后的阶段表示")
        )
        Edges = @(
            (New-Edge "g1" "g2"),
            (New-Edge "g2" "g4" "bottom" "top" "vertical"),
            (New-Edge "g3" "g4" "bottom" "top" "vertical"),
            (New-Edge "g4" "g5" "left" "right"),
            (New-Edge "g5" "g6" "bottom" "top" "vertical"),
            (New-Edge "g6" "g7")
        )
    },
    @{
        Name = "cnn_conformer_architecture"
        Width = 1460
        Height = 740
        Blocks = @(
            (New-Block "c1" 70 100 180 92 "对数梅尔频谱"),
            (New-Block "c2" 360 70 260 126 "无卷积前端时间分块输入`n每个标记覆盖全部频带`n沿时间轴按四帧切分"),
            (New-Block "c3" 720 100 200 92 "线性投影与归一化"),
            (New-Block "c4" 1040 50 260 146 "卷积增强Transformer编码器堆叠`n四层重复编码块`n相对位置注意力`n卷积核三十一"),
            (New-Block "c5" 1040 280 260 106 "末层序列输出"),
            (New-Block "c6" 720 460 220 106 "注意力池化`n压缩为句级表示"),
            (New-Block "c7" 400 470 160 92 "丢弃层"),
            (New-Block "c8" 140 470 180 92 "线性分类器"),
            (New-Block "c9" 140 610 180 92 "八类输出")
        )
        Edges = @(
            (New-Edge "c1" "c2"),
            (New-Edge "c2" "c3"),
            (New-Edge "c3" "c4"),
            (New-Edge "c4" "c5" "bottom" "top" "vertical"),
            (New-Edge "c5" "c6" "bottom" "top" "vertical"),
            (New-Edge "c6" "c7" "left" "right"),
            (New-Edge "c7" "c8" "left" "right"),
            (New-Edge "c8" "c9" "bottom" "top" "vertical")
        )
    },
    @{
        Name = "nostem_patch_tokenization"
        Width = 1400
        Height = 580
        Blocks = @(
            (New-Block "n1" 70 90 190 92 "输入频谱图`n频带数八十"),
            (New-Block "n2" 380 60 250 126 "整频带卷积切块`n卷积核高度等于全部频带`n卷积核宽度等于四帧"),
            (New-Block "n3" 760 60 230 126 "沿时间轴滑动`n步长等于四帧`n形成不重叠时间块"),
            (New-Block "n4" 760 280 230 126 "每个时间块映射为一个标记`n标记已包含全频带信息"),
            (New-Block "n5" 380 310 180 92 "层归一化"),
            (New-Block "n6" 120 310 180 92 "丢弃层"),
            (New-Block "n7" 120 450 220 92 "送入卷积增强Transformer序列编码")
        )
        Edges = @(
            (New-Edge "n1" "n2"),
            (New-Edge "n2" "n3"),
            (New-Edge "n3" "n4" "bottom" "top" "vertical"),
            (New-Edge "n4" "n5" "left" "right"),
            (New-Edge "n5" "n6" "left" "right"),
            (New-Edge "n6" "n7" "bottom" "top" "vertical")
        )
    },
    @{
        Name = "conformer_block"
        Width = 1460
        Height = 560
        Blocks = @(
            (New-Block "f1" 80 90 200 92 "输入序列"),
            (New-Block "f2" 380 70 240 126 "前馈模块一`n线性层`n激活函数`n线性层`n半残差缩放"),
            (New-Block "f3" 760 70 240 96 "相对位置多头自注意力"),
            (New-Block "f4" 1120 70 240 126 "卷积模块`n逐点卷积`n深度卷积`n门控与激活"),
            (New-Block "f5" 1120 320 240 126 "前馈模块二`n线性层`n激活函数`n线性层`n半残差缩放"),
            (New-Block "f6" 760 340 220 92 "LayerNorm"),
            (New-Block "f7" 380 330 220 92 "输出序列")
        )
        Edges = @(
            (New-Edge "f1" "f2"),
            (New-Edge "f2" "f3"),
            (New-Edge "f3" "f4"),
            (New-Edge "f4" "f5" "bottom" "top" "vertical"),
            (New-Edge "f5" "f6" "left" "right"),
            (New-Edge "f6" "f7" "left" "right")
        )
    }
)

foreach ($diagram in $diagrams) {
    $pngPath = Join-Path $OutputDir ($diagram.Name + ".png")
    $drawioPath = Join-Path $OutputDir ($diagram.Name + ".drawio")
    Render-PngDiagram -Diagram $diagram -PngPath $pngPath
    [System.IO.File]::WriteAllText($drawioPath, (Build-DrawioXml -Diagram $diagram), [System.Text.Encoding]::UTF8)
}

Write-Host "Generated diagrams in $OutputDir"

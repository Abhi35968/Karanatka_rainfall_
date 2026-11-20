$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $docxPath = "d:\PROJECT\karnataka_rainfall_cds\Stage9_2_Final_Author_Review_Final\Karnataka_Rainfall_Research_Paper_Final.docx"
    $pdfPath = "d:\PROJECT\karnataka_rainfall_cds\Stage9_2_Final_Author_Review_Final\Karnataka_Rainfall_Research_Paper_Final.pdf"
    $doc = $word.Documents.Open($docxPath)
    $wdFormatPDF = 17
    $doc.SaveAs([ref]$pdfPath, [ref]$wdFormatPDF)
    $doc.Close()
    Write-Output "PDF conversion successful"
}
finally {
    $word.Quit()
}


param([string]$file = '')
# Throwaway typing/viewing target for the benchmark: no persistence, has UIA text support, never touches user apps.
Add-Type -AssemblyName System.Windows.Forms
$f = New-Object System.Windows.Forms.Form
$f.Text = 'CuaBenchPad'
if ($file) { $f.Text = 'CuaBenchPad - ' + [IO.Path]::GetFileName($file) }
$f.Width = 500; $f.Height = 300
$t = New-Object System.Windows.Forms.TextBox; $t.Multiline = $true; $t.Dock = 'Fill'
if ($file -and (Test-Path $file)) { $t.Text = Get-Content -Raw $file }
$f.Controls.Add($t)
$f.Add_Shown({ $f.Activate(); $t.Focus() })
[void]$f.ShowDialog()

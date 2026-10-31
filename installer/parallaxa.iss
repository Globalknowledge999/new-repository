; Inno Setup script for Parallaxa-Setup.exe (built by .github/workflows/windows-installer.yml).
; The setup is small: it carries the Parallaxa source and installer scripts, then runs install.ps1,
; which downloads Python, the right PyTorch build and FFmpeg for this PC's GPU driver.

#ifndef AppVersion
  #define AppVersion "0.5.0"
#endif
#ifndef BuildNo
  #define BuildNo "local"
#endif

[Setup]
AppId={{58217CE4-5D28-4D24-8F70-FF33D8F2CF30}
AppName=Parallaxa
AppVersion={#AppVersion}
AppPublisher=Parallaxa (personal project)
DefaultDirName={localappdata}\Programs\Parallaxa
DefaultGroupName=Parallaxa
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
; The build number tells setups from different CI runs apart.
OutputBaseFilename=Parallaxa-Setup-{#AppVersion}-build{#BuildNo}
OutputDir=..\dist
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName=Parallaxa
UninstallDisplayIcon={app}\venv\Scripts\parallaxa-gui.exe
LicenseFile=..\LICENSE

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Shortcuts:"
Name: "models"; Description: "Download the depth models now (about 1.4 GB; otherwise on first use)"; GroupDescription: "Models:"

[Files]
Source: "install.ps1"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "uninstall.ps1"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "..\pyproject.toml"; DestDir: "{app}\installer\app"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}\installer\app"; Flags: ignoreversion
Source: "..\MODELS.md"; DestDir: "{app}\installer\app"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}\installer\app"; Flags: ignoreversion
Source: "..\parallaxa\*"; DestDir: "{app}\installer\app\parallaxa"; Excludes: "__pycache__,*.pyc"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\Parallaxa"; Filename: "{app}\venv\Scripts\parallaxa-gui.exe"; WorkingDir: "{userdocs}"
Name: "{group}\Uninstall Parallaxa"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Parallaxa"; Filename: "{app}\venv\Scripts\parallaxa-gui.exe"; WorkingDir: "{userdocs}"; Tasks: desktopicon

[Run]
; Visible console so the long PyTorch download shows progress. Shortcuts and the uninstall entry
; are handled by Inno Setup itself.
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\installer\install.ps1"" -InstallDir ""{app}"" -Source ""{app}\installer\app"" -NoShortcuts -NoRegister {code:ModelsFlag}"; \
  StatusMsg: "Downloading and installing Python, PyTorch and FFmpeg (several GB; this can take a while)..."; \
  Flags: waituntilterminated
Filename: "{app}\venv\Scripts\parallaxa-gui.exe"; Description: "Start Parallaxa"; Flags: postinstall nowait skipifsilent unchecked

[UninstallDelete]
Type: filesandordirs; Name: "{app}\venv"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\uv"
Type: filesandordirs; Name: "{app}\ffmpeg"
Type: filesandordirs; Name: "{app}\cache"
Type: filesandordirs; Name: "{app}\installer"
Type: files; Name: "{app}\install.log"
Type: files; Name: "{app}\uninstall.ps1"
Type: dirifempty; Name: "{app}"

[Code]
function ModelsFlag(Param: String): String;
begin
  if WizardIsTaskSelected('models') then Result := '' else Result := '-SkipModels';
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and not FileExists(ExpandConstant('{app}\venv\Scripts\parallaxa-gui.exe')) then
    MsgBox('The download step did not finish (see ' + ExpandConstant('{app}\install.log') + ').' + #13#10 +
           'Check your internet connection and run the setup again.', mbError, MB_OK);
end;

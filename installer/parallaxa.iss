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
; Installs to C:\Program Files\Parallaxa (asks for admin rights). The first page also offers
; "only for me", which needs no admin rights and installs to %LOCALAPPDATA%\Programs\Parallaxa.
DefaultDirName={autopf}\Parallaxa
DefaultGroupName=Parallaxa
DisableProgramGroupPage=yes
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog commandline
; Always ask "all users / only me". Inno's default (yes) silently reuses an earlier install's mode, which
; kept moving per-user installs from older builds back into %LOCALAPPDATA%.
UsePreviousPrivileges=no
; Don't reuse the folder of an earlier per-user install; that copy is removed instead (see [Code]).
UsePreviousAppDir=no
; The build number tells setups from different CI runs apart.
OutputBaseFilename=Parallaxa-Setup-{#AppVersion}-build{#BuildNo}
OutputDir=..\dist
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName=Parallaxa
UninstallDisplayIcon={app}\parallaxa.ico
SetupIconFile=..\parallaxa\assets\parallaxa.ico
LicenseFile=..\LICENSE

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Shortcuts:"
Name: "models"; Description: "Download the depth models now (about 1.4 GB; otherwise on first use)"; GroupDescription: "Models:"

[Files]
; The icon is copied straight into {app}: shortcuts are created before install.ps1 runs, so they must
; not point at a file that only appears once the Python environment exists.
Source: "..\parallaxa\assets\parallaxa.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "install.ps1"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "uninstall.ps1"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "..\pyproject.toml"; DestDir: "{app}\installer\app"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}\installer\app"; Flags: ignoreversion
Source: "..\MODELS.md"; DestDir: "{app}\installer\app"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}\installer\app"; Flags: ignoreversion
Source: "..\parallaxa\*"; DestDir: "{app}\installer\app\parallaxa"; Excludes: "__pycache__,*.pyc"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{autoprograms}\Parallaxa\Parallaxa"; Filename: "{app}\venv\Scripts\parallaxa-gui.exe"; WorkingDir: "{userdocs}"; IconFilename: "{app}\parallaxa.ico"
Name: "{autoprograms}\Parallaxa\Uninstall Parallaxa"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Parallaxa"; Filename: "{app}\venv\Scripts\parallaxa-gui.exe"; WorkingDir: "{userdocs}"; IconFilename: "{app}\parallaxa.ico"; Tasks: desktopicon

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
Type: files; Name: "{app}\parallaxa.ico"
Type: dirifempty; Name: "{app}"

[Code]
function ModelsFlag(Param: String): String;
begin
  if WizardIsTaskSelected('models') then Result := '' else Result := '-SkipModels';
end;

const
  UninstallKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{58217CE4-5D28-4D24-8F70-FF33D8F2CF30}_is1';

{ An all-users install replaces an earlier per-user one (in %LOCALAPPDATA%\Programs), so there aren't
  two copies. Downloaded models and settings live in the user's profile and are kept. }
procedure RemovePerUserCopy();
var
  Cmd, OldDir: String;
  Code, I: Integer;
begin
  if not IsAdminInstallMode then
    exit;
  OldDir := ExpandConstant('{localappdata}\Programs\Parallaxa');
  if RegQueryStringValue(HKCU, UninstallKey, 'UninstallString', Cmd) then
  begin
    WizardForm.StatusLabel.Caption := 'Removing the earlier per-user copy of Parallaxa...';
    Exec(RemoveQuotes(Cmd), '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART', '', SW_HIDE, ewWaitUntilTerminated, Code);
    { The uninstaller hands over to a temporary copy of itself and returns at once; give it time. }
    for I := 1 to 30 do
    begin
      if not DirExists(OldDir + '\venv') then
        break;
      Sleep(1000);
    end;
  end;
  { Clear whatever is left, but only if the folder is recognisably a Parallaxa install. }
  if DirExists(OldDir + '\venv') or DirExists(OldDir + '\installer') or FileExists(OldDir + '\install.log') then
    DelTree(OldDir, True, True, True);
  RegDeleteKeyIncludingSubkeys(HKCU, UninstallKey);
  RegDeleteKeyIncludingSubkeys(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\Parallaxa'); { script install }
  { Per-user shortcuts would otherwise sit next to the new all-users ones. }
  DeleteFile(ExpandConstant('{userdesktop}\Parallaxa.lnk'));
  DelTree(ExpandConstant('{userprograms}\Parallaxa'), True, True, True);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then
    RemovePerUserCopy();
  if (CurStep = ssPostInstall) and not FileExists(ExpandConstant('{app}\venv\Scripts\parallaxa-gui.exe')) then
    MsgBox('The download step did not finish (see ' + ExpandConstant('{app}\install.log') + ').' + #13#10 +
           'Check your internet connection and run the setup again.', mbError, MB_OK);
end;

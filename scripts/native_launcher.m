#import <AVFoundation/AVFoundation.h>
#import <Cocoa/Cocoa.h>

@interface GazeAppDelegate : NSObject <NSApplicationDelegate>
@property(nonatomic, strong) NSTask *task;
@property(nonatomic, strong) NSStatusItem *statusItem;
@property(nonatomic, strong) NSMenuItem *openItem;
@property(nonatomic, strong) NSMenuItem *hideItem;
@property(nonatomic, strong) NSMenuItem *enableItem;
@property(nonatomic, strong) NSMenuItem *disableItem;
@property(nonatomic, strong) NSFileHandle *logHandle;
@property(nonatomic, copy) NSString *projectPath;
@property(nonatomic, copy) NSString *pythonPath;
@property(nonatomic, copy) NSString *scriptPath;
@property(nonatomic, copy) NSString *logPath;
@property(nonatomic, copy) NSString *cacheDir;
@property(nonatomic, copy) NSString *controlPath;
@property(nonatomic) BOOL isQuitting;
@property(nonatomic) BOOL stopRequested;
@end

@implementation GazeAppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    [self configurePaths];
    [self openLog];
    [self createStatusItem];
    [self startCamera:nil];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return NO;
}

- (void)applicationWillTerminate:(NSNotification *)notification {
    self.isQuitting = YES;
    [self stopCameraProcess];
}

- (void)configurePaths {
    NSBundle *bundle = [NSBundle mainBundle];
    NSString *resourcePath = [bundle resourcePath];
    self.projectPath = [resourcePath stringByAppendingPathComponent:@"project"];
    self.pythonPath = [self.projectPath stringByAppendingPathComponent:@".venv/bin/python"];
    self.scriptPath = [self.projectPath stringByAppendingPathComponent:@"bin_single_window.py"];

    NSString *home = NSHomeDirectory();
    NSString *logDir = [home stringByAppendingPathComponent:@"Library/Logs"];
    NSString *supportDir = [[home stringByAppendingPathComponent:@"Library/Application Support"] stringByAppendingPathComponent:@"Gaze Correction Camera"];
    self.logPath = [logDir stringByAppendingPathComponent:@"Gaze Correction Camera.log"];
    self.cacheDir = [[home stringByAppendingPathComponent:@"Library/Caches"] stringByAppendingPathComponent:@"Gaze Correction Camera/matplotlib"];
    self.controlPath = [supportDir stringByAppendingPathComponent:@"control.txt"];

    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:logDir withIntermediateDirectories:YES attributes:nil error:nil];
    [fm createDirectoryAtPath:supportDir withIntermediateDirectories:YES attributes:nil error:nil];
    [fm createDirectoryAtPath:self.cacheDir withIntermediateDirectories:YES attributes:nil error:nil];
}

- (void)openLog {
    self.logHandle = [NSFileHandle fileHandleForWritingAtPath:self.logPath];
    if (!self.logHandle) {
        [@"" writeToFile:self.logPath atomically:YES encoding:NSUTF8StringEncoding error:nil];
        self.logHandle = [NSFileHandle fileHandleForWritingAtPath:self.logPath];
    }
    [self.logHandle seekToEndOfFile];
    [self logLine:[NSString stringWithFormat:@"\n--- Gaze Correction Camera native start: %@ ---", [NSDate date]]];
}

- (void)createStatusItem {
    self.statusItem = [[NSStatusBar systemStatusBar] statusItemWithLength:NSSquareStatusItemLength];
    NSStatusBarButton *button = self.statusItem.button;
    button.image = [self statusIcon];
    button.image.template = YES;
    button.toolTip = @"Gaze Correction Camera";

    NSMenu *menu = [[NSMenu alloc] initWithTitle:@"Gaze Correction Camera"];
    self.openItem = [[NSMenuItem alloc] initWithTitle:@"Show Window" action:@selector(showWindow:) keyEquivalent:@""];
    self.openItem.target = self;
    [menu addItem:self.openItem];

    self.hideItem = [[NSMenuItem alloc] initWithTitle:@"Hide Window" action:@selector(hideWindow:) keyEquivalent:@""];
    self.hideItem.target = self;
    [menu addItem:self.hideItem];

    [menu addItem:[NSMenuItem separatorItem]];
    self.enableItem = [[NSMenuItem alloc] initWithTitle:@"Enable Correction" action:@selector(enableCorrection:) keyEquivalent:@""];
    self.enableItem.target = self;
    [menu addItem:self.enableItem];

    self.disableItem = [[NSMenuItem alloc] initWithTitle:@"Disable Correction" action:@selector(disableCorrection:) keyEquivalent:@""];
    self.disableItem.target = self;
    [menu addItem:self.disableItem];

    [menu addItem:[NSMenuItem separatorItem]];
    NSMenuItem *quitItem = [[NSMenuItem alloc] initWithTitle:@"Quit" action:@selector(quitApp:) keyEquivalent:@""];
    quitItem.target = self;
    [menu addItem:quitItem];

    self.statusItem.menu = menu;
    [self updateMenuItems];
}

- (NSImage *)statusIcon {
    NSImage *image = [[NSImage alloc] initWithSize:NSMakeSize(18, 18)];
    [image lockFocus];

    [[NSColor clearColor] setFill];
    NSRectFill(NSMakeRect(0, 0, 18, 18));

    [[NSColor labelColor] setStroke];
    NSBezierPath *eye = [NSBezierPath bezierPath];
    [eye moveToPoint:NSMakePoint(2.2, 9.0)];
    [eye curveToPoint:NSMakePoint(9.0, 4.6) controlPoint1:NSMakePoint(4.2, 5.4) controlPoint2:NSMakePoint(6.4, 4.6)];
    [eye curveToPoint:NSMakePoint(15.8, 9.0) controlPoint1:NSMakePoint(11.6, 4.6) controlPoint2:NSMakePoint(13.8, 5.4)];
    [eye curveToPoint:NSMakePoint(9.0, 13.4) controlPoint1:NSMakePoint(13.8, 12.6) controlPoint2:NSMakePoint(11.6, 13.4)];
    [eye curveToPoint:NSMakePoint(2.2, 9.0) controlPoint1:NSMakePoint(6.4, 13.4) controlPoint2:NSMakePoint(4.2, 12.6)];
    eye.lineWidth = 1.6;
    [eye stroke];

    [[NSColor labelColor] setFill];
    NSBezierPath *pupil = [NSBezierPath bezierPathWithOvalInRect:NSMakeRect(6.4, 6.4, 5.2, 5.2)];
    [pupil fill];

    [image unlockFocus];
    image.template = YES;
    return image;
}

- (BOOL)cameraPermissionAllowed {
    AVAuthorizationStatus status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
    if (status == AVAuthorizationStatusNotDetermined) {
        dispatch_semaphore_t sema = dispatch_semaphore_create(0);
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^(BOOL granted) {
            dispatch_semaphore_signal(sema);
        }];
        dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);
        status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
    }

    if (status != AVAuthorizationStatusAuthorized) {
        [self showAlert:@"Camera access is blocked"
                message:@"Open System Settings > Privacy & Security > Camera and allow Gaze Correction Camera, then open it again from the menu bar icon."];
        return NO;
    }
    return YES;
}

- (void)startCamera:(id)sender {
    if (self.task && self.task.isRunning) {
        [self sendCommand:@"show"];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        [NSApp activateIgnoringOtherApps:YES];
        return;
    }

    if (![self cameraPermissionAllowed]) {
        [self enterMenuBarMode];
        return;
    }

    self.stopRequested = NO;
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    [NSApp activateIgnoringOtherApps:YES];

    NSTask *newTask = [[NSTask alloc] init];
    [newTask setCurrentDirectoryPath:self.projectPath];
    [newTask setLaunchPath:@"/usr/bin/arch"];
    [newTask setArguments:@[@"-arm64", self.pythonPath, self.scriptPath, @"--backend", @"mediapipe"]];

    NSMutableDictionary *env = [[[NSProcessInfo processInfo] environment] mutableCopy];
    env[@"PYTHONUNBUFFERED"] = @"1";
    env[@"PYTHONNOUSERSITE"] = @"1";
    env[@"MPLCONFIGDIR"] = self.cacheDir;
    env[@"GAZE_CONTROL_FILE"] = self.controlPath;
    [newTask setEnvironment:env];
    [newTask setStandardOutput:self.logHandle];
    [newTask setStandardError:self.logHandle];

    __weak GazeAppDelegate *weakSelf = self;
    newTask.terminationHandler = ^(NSTask *finishedTask) {
        dispatch_async(dispatch_get_main_queue(), ^{
            [weakSelf cameraTaskDidExit:finishedTask];
        });
    };

    @try {
        [self logLine:[NSString stringWithFormat:@"Launching camera: %@", [NSDate date]]];
        [self sendCommand:@"show"];
        [newTask launch];
        self.task = newTask;
        [self updateMenuItems];
    } @catch (NSException *exception) {
        NSString *message = [NSString stringWithFormat:@"Could not launch Python runtime: %@", [exception reason]];
        [self logLine:message];
        [self showAlert:@"Gaze Correction Camera could not start" message:message];
        self.task = nil;
        [self enterMenuBarMode];
    }
}

- (void)showWindow:(id)sender {
    [self startCamera:sender];
    [self sendCommand:@"show"];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    [NSApp activateIgnoringOtherApps:YES];
}

- (void)hideWindow:(id)sender {
    [self sendCommand:@"hide"];
    [self enterMenuBarMode];
}

- (void)enableCorrection:(id)sender {
    if (!(self.task && self.task.isRunning)) {
        [self startCamera:nil];
    }
    [self sendCommand:@"enable"];
}

- (void)disableCorrection:(id)sender {
    if (!(self.task && self.task.isRunning)) {
        [self startCamera:nil];
    }
    [self sendCommand:@"disable"];
}

- (void)quitApp:(id)sender {
    self.isQuitting = YES;
    [self sendCommand:@"quit"];
    [self stopCameraProcess];
    [NSApp terminate:nil];
}

- (void)stopCameraProcess {
    if (self.task && self.task.isRunning) {
        [self.task terminate];
    }
}

- (void)cameraTaskDidExit:(NSTask *)finishedTask {
    int exitCode = finishedTask.terminationStatus;
    BOOL expectedExit = self.stopRequested || self.isQuitting || exitCode == 0;
    self.task = nil;
    self.stopRequested = NO;
    [self updateMenuItems];

    if (self.isQuitting) {
        return;
    }

    [self logLine:[NSString stringWithFormat:@"Camera process exited with code %d", exitCode]];
    [self enterMenuBarMode];

    if (!expectedExit) {
        [self showAlert:@"Gaze Correction Camera could not start"
                message:@"Check the log at ~/Library/Logs/Gaze Correction Camera.log. The app will stay in the menu bar so you can try opening it again."];
    }
}

- (void)enterMenuBarMode {
    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
    [self updateMenuItems];
}

- (void)updateMenuItems {
    BOOL running = self.task && self.task.isRunning;
    self.openItem.title = running ? @"Show Window" : @"Open Camera";
    self.hideItem.enabled = running;
    self.enableItem.enabled = running;
    self.disableItem.enabled = running;
}

- (void)sendCommand:(NSString *)command {
    if (!self.controlPath) {
        return;
    }
    NSString *payload = [NSString stringWithFormat:@"%@\n%.6f\n", command, [[NSDate date] timeIntervalSince1970]];
    [payload writeToFile:self.controlPath atomically:YES encoding:NSUTF8StringEncoding error:nil];
    [self logLine:[NSString stringWithFormat:@"Command: %@", command]];
}

- (void)showAlert:(NSString *)title message:(NSString *)message {
    NSAlert *alert = [[NSAlert alloc] init];
    alert.messageText = title;
    alert.informativeText = message;
    [alert addButtonWithTitle:@"OK"];
    [alert runModal];
}

- (void)logLine:(NSString *)line {
    if (!self.logHandle) {
        return;
    }
    NSString *text = [line stringByAppendingString:@"\n"];
    [self.logHandle writeData:[text dataUsingEncoding:NSUTF8StringEncoding]];
}

@end

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        GazeAppDelegate *delegate = [[GazeAppDelegate alloc] init];
        app.delegate = delegate;
        [app run];
        return 0;
    }
}

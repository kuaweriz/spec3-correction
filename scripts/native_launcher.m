#import <AVFoundation/AVFoundation.h>
#import <Cocoa/Cocoa.h>

@interface GazeAppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@property(nonatomic, strong) NSTask *task;
@property(nonatomic, strong) NSStatusItem *statusItem;
@property(nonatomic, strong) NSMenuItem *openItem;
@property(nonatomic, strong) NSMenuItem *hideItem;
@property(nonatomic, strong) NSMenuItem *enableItem;
@property(nonatomic, strong) NSMenuItem *disableItem;
@property(nonatomic, strong) NSFileHandle *logHandle;
@property(nonatomic, strong) NSWindow *logWindow;
@property(nonatomic, strong) NSTextView *logTextView;
@property(nonatomic, strong) NSTextField *logStatusLabel;
@property(nonatomic, strong) NSTimer *logRefreshTimer;
@property(nonatomic, copy) NSString *projectPath;
@property(nonatomic, copy) NSString *pythonPath;
@property(nonatomic, copy) NSString *scriptPath;
@property(nonatomic, copy) NSString *logPath;
@property(nonatomic, copy) NSString *cacheDir;
@property(nonatomic, copy) NSString *controlPath;
@property(nonatomic, copy) NSString *cameraListPath;
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
    [self.logRefreshTimer invalidate];
    [self stopCameraProcess];
}

- (void)application:(NSApplication *)application openURLs:(NSArray<NSURL *> *)urls {
    for (NSURL *url in urls) {
        if ([[url scheme] isEqualToString:@"spec3correction"] && [[[url host] lowercaseString] isEqualToString:@"logs"]) {
            [self showLogWindow:nil];
            return;
        }
    }
}

- (void)configurePaths {
    NSBundle *bundle = [NSBundle mainBundle];
    NSString *resourcePath = [bundle resourcePath];
    self.projectPath = [resourcePath stringByAppendingPathComponent:@"project"];
    self.pythonPath = [self.projectPath stringByAppendingPathComponent:@".venv/bin/python"];
    self.scriptPath = [self.projectPath stringByAppendingPathComponent:@"bin_single_window.py"];

    NSString *home = NSHomeDirectory();
    NSString *logDir = [home stringByAppendingPathComponent:@"Library/Logs"];
    NSString *supportDir = [[home stringByAppendingPathComponent:@"Library/Application Support"] stringByAppendingPathComponent:@"spec3 correction"];
    self.logPath = [logDir stringByAppendingPathComponent:@"spec3 correction.log"];
    self.cacheDir = [[home stringByAppendingPathComponent:@"Library/Caches"] stringByAppendingPathComponent:@"spec3 correction/matplotlib"];
    self.controlPath = [supportDir stringByAppendingPathComponent:@"control.txt"];
    self.cameraListPath = [supportDir stringByAppendingPathComponent:@"cameras.json"];

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
    [self logLine:[NSString stringWithFormat:@"\n--- spec3 correction native start: %@ ---", [NSDate date]]];
}

- (void)createStatusItem {
    self.statusItem = [[NSStatusBar systemStatusBar] statusItemWithLength:NSSquareStatusItemLength];
    NSStatusBarButton *button = self.statusItem.button;
    button.image = [self statusIcon];
    button.image.template = YES;
    button.toolTip = @"spec3 correction";

    NSMenu *menu = [[NSMenu alloc] initWithTitle:@"spec3 correction"];
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
    NSMenuItem *logsItem = [[NSMenuItem alloc] initWithTitle:@"Open Logs" action:@selector(openLogs:) keyEquivalent:@""];
    logsItem.target = self;
    [menu addItem:logsItem];

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
                message:@"Open System Settings > Privacy & Security > Camera and allow spec3 correction, then open it again from the menu bar icon."];
        return NO;
    }
    return YES;
}

- (void)writeCameraList {
    NSArray<AVCaptureDevice *> *devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    NSMutableArray *items = [NSMutableArray arrayWithCapacity:devices.count];
    NSInteger index = 0;
    for (AVCaptureDevice *device in devices) {
        NSString *name = device.localizedName ?: [NSString stringWithFormat:@"Camera %ld", (long)index];
        NSString *uniqueID = device.uniqueID ?: @"";
        NSString *modelID = device.modelID ?: @"";
        [items addObject:@{
            @"id": @(index),
            @"name": name,
            @"unique_id": uniqueID,
            @"model_id": modelID,
        }];
        index += 1;
    }

    NSData *data = [NSJSONSerialization dataWithJSONObject:items options:NSJSONWritingPrettyPrinted error:nil];
    if (data) {
        [data writeToFile:self.cameraListPath atomically:YES];
    }
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
    [self writeCameraList];

    [[NSFileManager defaultManager] removeItemAtPath:self.controlPath error:nil];
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
    env[@"SPEC3_SUPPORT_DIR"] = [self.controlPath stringByDeletingLastPathComponent];
    env[@"SPEC3_CAMERA_LIST_FILE"] = self.cameraListPath;
    env[@"SPEC3_LOG_FILE"] = self.logPath;
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
        [self showAlert:@"spec3 correction could not start" message:message];
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

- (void)openLogs:(id)sender {
    [self showLogWindow:sender];
}

- (void)showLogWindow:(id)sender {
    if (!self.logWindow) {
        [self createLogWindow];
    }

    [self refreshLogWindow:nil];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    [NSApp activateIgnoringOtherApps:YES];
    [self.logWindow makeKeyAndOrderFront:nil];

    [self.logRefreshTimer invalidate];
    self.logRefreshTimer = [NSTimer scheduledTimerWithTimeInterval:1.0
                                                            target:self
                                                          selector:@selector(refreshLogWindow:)
                                                          userInfo:nil
                                                           repeats:YES];
}

- (void)createLogWindow {
    NSRect frame = NSMakeRect(0, 0, 820, 560);
    NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;
    self.logWindow = [[NSWindow alloc] initWithContentRect:frame
                                                 styleMask:style
                                                   backing:NSBackingStoreBuffered
                                                     defer:NO];
    self.logWindow.title = @"spec3 correction logs";
    self.logWindow.releasedWhenClosed = NO;
    self.logWindow.delegate = self;
    self.logWindow.backgroundColor = [NSColor colorWithCalibratedRed:0.08 green:0.07 blue:0.07 alpha:1.0];
    [self.logWindow center];

    NSView *content = self.logWindow.contentView;

    NSTextField *title = [NSTextField labelWithString:@"Logs"];
    title.frame = NSMakeRect(24, 510, 260, 28);
    title.font = [NSFont systemFontOfSize:24 weight:NSFontWeightSemibold];
    title.textColor = [NSColor colorWithCalibratedRed:0.98 green:0.95 blue:0.91 alpha:1.0];
    title.autoresizingMask = NSViewMinYMargin;
    [content addSubview:title];

    self.logStatusLabel = [NSTextField labelWithString:self.logPath];
    self.logStatusLabel.frame = NSMakeRect(25, 486, 520, 20);
    self.logStatusLabel.font = [NSFont systemFontOfSize:12 weight:NSFontWeightRegular];
    self.logStatusLabel.textColor = [NSColor colorWithCalibratedRed:0.62 green:0.58 blue:0.55 alpha:1.0];
    self.logStatusLabel.lineBreakMode = NSLineBreakByTruncatingMiddle;
    self.logStatusLabel.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
    [content addSubview:self.logStatusLabel];

    NSButton *refreshButton = [NSButton buttonWithTitle:@"Refresh" target:self action:@selector(refreshLogWindow:)];
    refreshButton.frame = NSMakeRect(548, 500, 86, 32);
    refreshButton.bezelStyle = NSBezelStyleRounded;
    refreshButton.autoresizingMask = NSViewMinXMargin | NSViewMinYMargin;
    [content addSubview:refreshButton];

    NSButton *copyButton = [NSButton buttonWithTitle:@"Copy Path" target:self action:@selector(copyLogPath:)];
    copyButton.frame = NSMakeRect(640, 500, 88, 32);
    copyButton.bezelStyle = NSBezelStyleRounded;
    copyButton.autoresizingMask = NSViewMinXMargin | NSViewMinYMargin;
    [content addSubview:copyButton];

    NSButton *fileButton = [NSButton buttonWithTitle:@"Open File" target:self action:@selector(openRawLogFile:)];
    fileButton.frame = NSMakeRect(734, 500, 78, 32);
    fileButton.bezelStyle = NSBezelStyleRounded;
    fileButton.autoresizingMask = NSViewMinXMargin | NSViewMinYMargin;
    [content addSubview:fileButton];

    NSScrollView *scrollView = [[NSScrollView alloc] initWithFrame:NSMakeRect(24, 24, 772, 448)];
    scrollView.borderType = NSNoBorder;
    scrollView.hasVerticalScroller = YES;
    scrollView.autohidesScrollers = YES;
    scrollView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    scrollView.drawsBackground = YES;
    scrollView.backgroundColor = [NSColor colorWithCalibratedRed:0.12 green:0.10 blue:0.10 alpha:1.0];

    self.logTextView = [[NSTextView alloc] initWithFrame:scrollView.contentView.bounds];
    self.logTextView.editable = NO;
    self.logTextView.selectable = YES;
    self.logTextView.drawsBackground = YES;
    self.logTextView.backgroundColor = [NSColor colorWithCalibratedRed:0.12 green:0.10 blue:0.10 alpha:1.0];
    self.logTextView.textColor = [NSColor colorWithCalibratedRed:0.88 green:0.86 blue:0.82 alpha:1.0];
    self.logTextView.font = [NSFont monospacedSystemFontOfSize:12.0 weight:NSFontWeightRegular];
    self.logTextView.textContainerInset = NSMakeSize(14, 12);
    self.logTextView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.logTextView.textContainer.widthTracksTextView = YES;
    self.logTextView.textContainer.containerSize = NSMakeSize(scrollView.contentView.bounds.size.width, CGFLOAT_MAX);

    scrollView.documentView = self.logTextView;
    [content addSubview:scrollView];
}

- (void)refreshLogWindow:(id)sender {
    if (!self.logTextView) {
        return;
    }

    NSString *text = [self readLogPreview];
    [self.logTextView.textStorage setAttributedString:[self styledLogText:text]];
    [self.logTextView scrollRangeToVisible:NSMakeRange(self.logTextView.string.length, 0)];
    self.logStatusLabel.stringValue = [self logStatusText];
}

- (NSString *)readLogPreview {
    NSFileHandle *reader = [NSFileHandle fileHandleForReadingAtPath:self.logPath];
    if (!reader) {
        return @"Log file is not created yet.";
    }

    unsigned long long size = [reader seekToEndOfFile];
    NSUInteger maxBytes = 256 * 1024;
    NSString *prefix = @"";
    if (size > maxBytes) {
        [reader seekToFileOffset:(size - maxBytes)];
        prefix = @"Showing the last 256 KB of the log.\n\n";
    } else {
        [reader seekToFileOffset:0];
    }

    NSData *data = [reader readDataToEndOfFile];
    [reader closeFile];

    NSString *body = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    if (!body) {
        body = [[NSString alloc] initWithData:data encoding:NSISOLatin1StringEncoding] ?: @"Could not decode log text.";
    }
    if (body.length == 0) {
        body = @"Log is empty.";
    }
    return [prefix stringByAppendingString:body];
}

- (NSAttributedString *)styledLogText:(NSString *)text {
    NSMutableAttributedString *result = [[NSMutableAttributedString alloc] init];
    NSFont *font = [NSFont monospacedSystemFontOfSize:12.0 weight:NSFontWeightRegular];
    NSMutableParagraphStyle *paragraph = [[NSMutableParagraphStyle alloc] init];
    paragraph.lineSpacing = 2.0;

    NSColor *normal = [NSColor colorWithCalibratedRed:0.86 green:0.84 blue:0.80 alpha:1.0];
    NSColor *muted = [NSColor colorWithCalibratedRed:0.58 green:0.56 blue:0.54 alpha:1.0];
    NSColor *blue = [NSColor colorWithCalibratedRed:0.48 green:0.73 blue:1.00 alpha:1.0];
    NSColor *green = [NSColor colorWithCalibratedRed:0.54 green:0.86 blue:0.57 alpha:1.0];
    NSColor *orange = [NSColor colorWithCalibratedRed:1.00 green:0.62 blue:0.28 alpha:1.0];
    NSColor *red = [NSColor colorWithCalibratedRed:1.00 green:0.38 blue:0.43 alpha:1.0];

    for (NSString *line in [text componentsSeparatedByString:@"\n"]) {
        NSString *lower = [line lowercaseString];
        NSColor *color = normal;
        if ([lower containsString:@"error"] || [lower containsString:@"failed"] ||
            [lower containsString:@"could not"] || [lower containsString:@"traceback"] ||
            [lower containsString:@"exception"]) {
            color = red;
        } else if ([lower containsString:@"warning"] || [lower containsString:@"deprecated"]) {
            color = orange;
        } else if ([lower containsString:@"command:"] || [lower containsString:@"camera"]) {
            color = blue;
        } else if ([lower containsString:@"launching"] || [lower containsString:@"native start"] ||
                   [lower containsString:@"shutdown complete"]) {
            color = green;
        } else if (line.length == 0 || [lower containsString:@"showing the last"]) {
            color = muted;
        }

        NSDictionary *attrs = @{
            NSFontAttributeName: font,
            NSForegroundColorAttributeName: color,
            NSParagraphStyleAttributeName: paragraph,
        };
        NSString *lineWithBreak = [line stringByAppendingString:@"\n"];
        [result appendAttributedString:[[NSAttributedString alloc] initWithString:lineWithBreak attributes:attrs]];
    }
    return result;
}

- (NSString *)logStatusText {
    NSDictionary *attrs = [[NSFileManager defaultManager] attributesOfItemAtPath:self.logPath error:nil];
    NSNumber *sizeNumber = attrs[NSFileSize];
    NSDate *modified = attrs[NSFileModificationDate];
    double kb = sizeNumber ? sizeNumber.doubleValue / 1024.0 : 0.0;
    NSString *timeText = @"not updated yet";
    if (modified) {
        NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
        formatter.dateStyle = NSDateFormatterNoStyle;
        formatter.timeStyle = NSDateFormatterMediumStyle;
        timeText = [formatter stringFromDate:modified];
    }
    return [NSString stringWithFormat:@"%@  •  %.1f KB  •  updated %@", self.logPath, kb, timeText];
}

- (void)copyLogPath:(id)sender {
    NSPasteboard *pasteboard = [NSPasteboard generalPasteboard];
    [pasteboard clearContents];
    [pasteboard setString:self.logPath forType:NSPasteboardTypeString];
    self.logStatusLabel.stringValue = @"Log path copied to clipboard";
}

- (void)openRawLogFile:(id)sender {
    NSURL *logURL = [NSURL fileURLWithPath:self.logPath];
    [[NSWorkspace sharedWorkspace] openURL:logURL];
}

- (void)windowWillClose:(NSNotification *)notification {
    if (notification.object == self.logWindow) {
        [self.logRefreshTimer invalidate];
        self.logRefreshTimer = nil;
    }
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
        [self showAlert:@"spec3 correction could not start"
                message:@"Check the log at ~/Library/Logs/spec3 correction.log. The app will stay in the menu bar so you can try opening it again."];
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

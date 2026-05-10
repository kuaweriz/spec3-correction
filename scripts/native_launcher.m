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
@property(nonatomic, strong) NSWindow *trainingWindow;
@property(nonatomic, strong) NSTextField *trainingTitleLabel;
@property(nonatomic, strong) NSTextField *trainingHintLabel;
@property(nonatomic, strong) NSTextField *trainingStatusLabel;
@property(nonatomic, strong) NSTextField *trainingModelLabel;
@property(nonatomic, strong) NSTextField *trainingReadLabel;
@property(nonatomic, strong) NSTextField *trainingLiveLabel;
@property(nonatomic, strong) NSTextField *trainingLookLabel;
@property(nonatomic, strong) NSTextField *trainingReadGuideLabel;
@property(nonatomic, strong) NSTextField *trainingLiveGuideLabel;
@property(nonatomic, strong) NSTextField *trainingLookGuideLabel;
@property(nonatomic, strong) NSTextField *trainingNeedLabel;
@property(nonatomic, strong) NSTextField *trainingQualityLabel;
@property(nonatomic, strong) NSTextField *trainingPoolLabel;
@property(nonatomic, strong) NSTextField *trainingRecordingLabel;
@property(nonatomic, strong) NSTextField *trainingNextLabel;
@property(nonatomic, strong) NSProgressIndicator *trainingReadProgress;
@property(nonatomic, strong) NSProgressIndicator *trainingLiveProgress;
@property(nonatomic, strong) NSProgressIndicator *trainingLookProgress;
@property(nonatomic, strong) NSButton *trainingReadButton;
@property(nonatomic, strong) NSButton *trainingLiveButton;
@property(nonatomic, strong) NSButton *trainingLookButton;
@property(nonatomic, strong) NSButton *trainingStopButton;
@property(nonatomic, strong) NSButton *trainingResetButton;
@property(nonatomic, strong) NSButton *trainingFolderButton;
@property(nonatomic, strong) NSButton *trainingTrainButton;
@property(nonatomic, strong) NSTimer *trainingRefreshTimer;
@property(nonatomic, strong) NSWindow *settingsWindow;
@property(nonatomic, strong) NSTextField *settingsStatusLabel;
@property(nonatomic, strong) NSTextField *settingsTitleLabel;
@property(nonatomic, strong) NSTextField *settingsThemeLabel;
@property(nonatomic, strong) NSTextField *settingsStyleLabel;
@property(nonatomic, strong) NSTextField *settingsFontLabel;
@property(nonatomic, strong) NSTextField *settingsLanguageLabel;
@property(nonatomic, strong) NSPopUpButton *themePopup;
@property(nonatomic, strong) NSPopUpButton *stylePopup;
@property(nonatomic, strong) NSPopUpButton *fontPopup;
@property(nonatomic, strong) NSPopUpButton *languagePopup;
@property(nonatomic, strong) NSButton *settingsSaveButton;
@property(nonatomic, copy) NSString *projectPath;
@property(nonatomic, copy) NSString *pythonPath;
@property(nonatomic, copy) NSString *scriptPath;
@property(nonatomic, copy) NSString *logPath;
@property(nonatomic, copy) NSString *cacheDir;
@property(nonatomic, copy) NSString *controlPath;
@property(nonatomic, copy) NSString *nativeRequestPath;
@property(nonatomic, copy) NSString *cameraListPath;
@property(nonatomic, copy) NSString *trainingStatePath;
@property(nonatomic, copy) NSString *preferencesPath;
@property(nonatomic, strong) NSTimer *nativeRequestTimer;
@property(nonatomic) NSTimeInterval lastNativeRequestMTime;
@property(nonatomic) BOOL isQuitting;
@property(nonatomic) BOOL stopRequested;
@end

@implementation GazeAppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    [self configurePaths];
    [self openLog];
    [self createStatusItem];
    self.nativeRequestTimer = [NSTimer scheduledTimerWithTimeInterval:0.20
                                                               target:self
                                                             selector:@selector(processNativeRequest:)
                                                             userInfo:nil
                                                              repeats:YES];
    [self startCamera:nil];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return NO;
}

- (void)applicationWillTerminate:(NSNotification *)notification {
    self.isQuitting = YES;
    [self.logRefreshTimer invalidate];
    [self.trainingRefreshTimer invalidate];
    [self.nativeRequestTimer invalidate];
    [self stopCameraProcess];
}

- (void)application:(NSApplication *)application openURLs:(NSArray<NSURL *> *)urls {
    for (NSURL *url in urls) {
        if ([[url scheme] isEqualToString:@"spec3correction"] && [[[url host] lowercaseString] isEqualToString:@"logs"]) {
            [self showLogWindow:nil];
            return;
        }
        if ([[url scheme] isEqualToString:@"spec3correction"] && [[[url host] lowercaseString] isEqualToString:@"training"]) {
            [self showTrainingWindow:nil];
            return;
        }
        if ([[url scheme] isEqualToString:@"spec3correction"] && [[[url host] lowercaseString] isEqualToString:@"settings"]) {
            [self showSettingsWindow:nil];
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
    self.nativeRequestPath = [supportDir stringByAppendingPathComponent:@"native_request.txt"];
    self.cameraListPath = [supportDir stringByAppendingPathComponent:@"cameras.json"];
    self.trainingStatePath = [supportDir stringByAppendingPathComponent:@"training_state.json"];
    self.preferencesPath = [supportDir stringByAppendingPathComponent:@"preferences.json"];

    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:logDir withIntermediateDirectories:YES attributes:nil error:nil];
    [fm createDirectoryAtPath:supportDir withIntermediateDirectories:YES attributes:nil error:nil];
    [fm createDirectoryAtPath:self.cacheDir withIntermediateDirectories:YES attributes:nil error:nil];
    NSDictionary *requestAttrs = [fm attributesOfItemAtPath:self.nativeRequestPath error:nil];
    NSDate *requestDate = requestAttrs[NSFileModificationDate];
    self.lastNativeRequestMTime = requestDate ? [requestDate timeIntervalSince1970] : 0.0;
    [self ensurePreferencesFile];
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

- (void)ensurePreferencesFile {
    if ([[NSFileManager defaultManager] fileExistsAtPath:self.preferencesPath]) {
        return;
    }
    [self writePreferencesWithTheme:@"dark" language:@"en" style:@"orange" font:@"rounded"];
}

- (NSDictionary *)readPreferences {
    NSData *data = [NSData dataWithContentsOfFile:self.preferencesPath];
    if (!data) {
        return @{@"theme": @"dark", @"language": @"en", @"style": @"orange", @"font": @"rounded"};
    }
    id json = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    if (![json isKindOfClass:[NSDictionary class]]) {
        return @{@"theme": @"dark", @"language": @"en", @"style": @"orange", @"font": @"rounded"};
    }
    return (NSDictionary *)json;
}

- (NSString *)preferenceValue:(NSString *)key fallback:(NSString *)fallback {
    id value = [self readPreferences][key];
    return [value isKindOfClass:[NSString class]] ? value : fallback;
}

- (BOOL)isLightTheme {
    return [[self preferenceValue:@"theme" fallback:@"dark"] isEqualToString:@"light"];
}

- (BOOL)isRussian {
    return [[self preferenceValue:@"language" fallback:@"en"] isEqualToString:@"ru"];
}

- (NSString *)textEN:(NSString *)en ru:(NSString *)ru {
    return [self isRussian] ? ru : en;
}

- (void)writePreferencesWithTheme:(NSString *)theme language:(NSString *)language style:(NSString *)style font:(NSString *)font {
    NSDictionary *payload = @{
        @"theme": theme ?: @"dark",
        @"language": language ?: @"en",
        @"style": style ?: @"orange",
        @"font": font ?: @"rounded",
    };
    NSData *data = [NSJSONSerialization dataWithJSONObject:payload options:NSJSONWritingPrettyPrinted error:nil];
    if (data) {
        [data writeToFile:self.preferencesPath atomically:YES];
    }
}

- (NSString *)selectedStyleKey {
    NSString *value = [self preferenceValue:@"style" fallback:@"orange"];
    NSSet *allowed = [NSSet setWithArray:@[@"orange", @"blue", @"mint", @"violet", @"graphite"]];
    return [allowed containsObject:value] ? value : @"orange";
}

- (NSString *)selectedFontKey {
    NSString *value = [self preferenceValue:@"font" fallback:@"rounded"];
    NSSet *allowed = [NSSet setWithArray:@[@"rounded", @"system", @"compact", @"mono", @"serif"]];
    return [allowed containsObject:value] ? value : @"rounded";
}

- (NSColor *)accentColor {
    NSString *style = [self selectedStyleKey];
    if ([style isEqualToString:@"blue"]) {
        return [NSColor colorWithCalibratedRed:0.22 green:0.53 blue:1.00 alpha:1.0];
    }
    if ([style isEqualToString:@"mint"]) {
        return [NSColor colorWithCalibratedRed:0.22 green:0.78 blue:0.52 alpha:1.0];
    }
    if ([style isEqualToString:@"violet"]) {
        return [NSColor colorWithCalibratedRed:0.68 green:0.44 blue:1.00 alpha:1.0];
    }
    if ([style isEqualToString:@"graphite"]) {
        return [NSColor colorWithCalibratedRed:0.64 green:0.66 blue:0.69 alpha:1.0];
    }
    return [NSColor colorWithCalibratedRed:1.00 green:0.58 blue:0.27 alpha:1.0];
}

- (NSColor *)windowBackgroundColor {
    if ([self isLightTheme]) {
        return [NSColor colorWithCalibratedRed:0.94 green:0.95 blue:0.97 alpha:1.0];
    }
    return [NSColor colorWithCalibratedRed:0.08 green:0.07 blue:0.07 alpha:1.0];
}

- (NSColor *)primaryTextColor {
    if ([self isLightTheme]) {
        return [NSColor colorWithCalibratedRed:0.10 green:0.11 blue:0.13 alpha:1.0];
    }
    return [NSColor colorWithCalibratedRed:0.98 green:0.95 blue:0.91 alpha:1.0];
}

- (NSColor *)secondaryTextColor {
    if ([self isLightTheme]) {
        return [NSColor colorWithCalibratedRed:0.36 green:0.38 blue:0.43 alpha:1.0];
    }
    return [NSColor colorWithCalibratedRed:0.64 green:0.60 blue:0.56 alpha:1.0];
}

- (NSAppearance *)preferredAppearance {
    return [NSAppearance appearanceNamed:([self isLightTheme] ? NSAppearanceNameAqua : NSAppearanceNameDarkAqua)];
}

- (void)applyAppearanceToWindow:(NSWindow *)window {
    if (window) {
        window.appearance = [self preferredAppearance];
    }
}

- (NSFont *)uiFontOfSize:(CGFloat)size weight:(NSFontWeight)weight {
    NSString *fontChoice = [self selectedFontKey];
    if ([fontChoice isEqualToString:@"mono"]) {
        return [NSFont monospacedSystemFontOfSize:size weight:weight];
    }
    if ([fontChoice isEqualToString:@"serif"]) {
        NSString *name = weight >= NSFontWeightSemibold ? @"NewYork-Bold" : @"NewYork-Regular";
        return [NSFont fontWithName:name size:size] ?: ([NSFont fontWithName:@"Times New Roman" size:size] ?: [NSFont systemFontOfSize:size weight:weight]);
    }
    NSFont *font = [NSFont systemFontOfSize:size weight:weight];
    if ([fontChoice isEqualToString:@"system"]) {
        return font;
    }
    if ([fontChoice isEqualToString:@"compact"]) {
        NSFont *compact = [NSFont fontWithName:@"SF Compact" size:size];
        if (compact) {
            return compact;
        }
    }
    NSFontDescriptor *rounded = [[font fontDescriptor] fontDescriptorWithDesign:NSFontDescriptorSystemDesignRounded];
    return rounded ? ([NSFont fontWithDescriptor:rounded size:size] ?: font) : font;
}

- (void)styleButton:(NSButton *)button title:(NSString *)title emphasized:(BOOL)emphasized {
    button.appearance = [self preferredAppearance];
    NSColor *textColor = [self isLightTheme]
        ? [NSColor colorWithCalibratedRed:0.08 green:0.09 blue:0.11 alpha:1.0]
        : [NSColor colorWithCalibratedRed:0.96 green:0.94 blue:0.90 alpha:1.0];
    if (emphasized) {
        textColor = [self accentColor];
    }
    NSDictionary *attrs = @{
        NSForegroundColorAttributeName: textColor,
        NSFontAttributeName: [self uiFontOfSize:13 weight:NSFontWeightMedium],
    };
    button.attributedTitle = [[NSAttributedString alloc] initWithString:title attributes:attrs];
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
    NSMenuItem *aiTrainingItem = [[NSMenuItem alloc] initWithTitle:@"Open Training Panel" action:@selector(showTrainingWindow:) keyEquivalent:@""];
    aiTrainingItem.target = self;
    [menu addItem:aiTrainingItem];

    NSMenuItem *settingsItem = [[NSMenuItem alloc] initWithTitle:@"Settings" action:@selector(showSettingsWindow:) keyEquivalent:@""];
    settingsItem.target = self;
    [menu addItem:settingsItem];

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
    env[@"SPEC3_NATIVE_REQUEST_FILE"] = self.nativeRequestPath;
    env[@"SPEC3_SUPPORT_DIR"] = [self.controlPath stringByDeletingLastPathComponent];
    env[@"SPEC3_CAMERA_LIST_FILE"] = self.cameraListPath;
    env[@"SPEC3_LOG_FILE"] = self.logPath;
    env[@"SPEC3_TRAINING_STATE_FILE"] = self.trainingStatePath;
    env[@"SPEC3_PREFERENCES_FILE"] = self.preferencesPath;
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

- (void)recordReadSamples:(id)sender {
    if (!(self.task && self.task.isRunning)) {
        [self startCamera:nil];
    }
    [self sendCommand:@"record_read"];
    [self refreshTrainingWindowSoon];
}

- (void)recordLiveSamples:(id)sender {
    if (!(self.task && self.task.isRunning)) {
        [self startCamera:nil];
    }
    [self sendCommand:@"record_live"];
    [self refreshTrainingWindowSoon];
}

- (void)recordGlanceSamples:(id)sender {
    if (!(self.task && self.task.isRunning)) {
        [self startCamera:nil];
    }
    [self sendCommand:@"record_glance"];
    [self refreshTrainingWindowSoon];
}

- (void)stopRecordingSamples:(id)sender {
    [self sendCommand:@"record_stop"];
    [self refreshTrainingWindowSoon];
}

- (void)trainPersonalAI:(id)sender {
    if (!(self.task && self.task.isRunning)) {
        [self startCamera:nil];
    }
    [self sendCommand:@"train_ai"];
    [self refreshTrainingWindowSoon];
}

- (void)resetTrainingSamples:(id)sender {
    NSAlert *alert = [[NSAlert alloc] init];
    alert.messageText = [self textEN:@"Reset current samples?" ru:@"Обнулить текущие записи?"];
    alert.informativeText = [self textEN:@"This clears only the new samples in this session. The trained model and saved training history stay in place." ru:@"Это удалит только новые записи текущей сессии. Обученная модель и сохранённая история обучения останутся."];
    [alert addButtonWithTitle:[self textEN:@"Reset" ru:@"Обнулить"]];
    [alert addButtonWithTitle:[self textEN:@"Cancel" ru:@"Отмена"]];
    if ([alert runModal] != NSAlertFirstButtonReturn) {
        return;
    }
    [self sendCommand:@"reset_training"];
    [self refreshTrainingWindowSoon];
}

- (void)openTrainingDataFolder:(id)sender {
    NSString *supportDir = [self.trainingStatePath stringByDeletingLastPathComponent];
    NSURL *folderURL = [NSURL fileURLWithPath:supportDir];
    [[NSWorkspace sharedWorkspace] openURL:folderURL];
}

- (void)showTrainingWindow:(id)sender {
    if (!(self.task && self.task.isRunning)) {
        [self startCamera:nil];
    }
    if (!self.trainingWindow) {
        [self createTrainingWindow];
    }

    [self refreshTrainingWindow:nil];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    [NSApp activateIgnoringOtherApps:YES];
    [self.trainingWindow makeKeyAndOrderFront:nil];

    [self.trainingRefreshTimer invalidate];
    self.trainingRefreshTimer = [NSTimer scheduledTimerWithTimeInterval:0.5
                                                                 target:self
                                                               selector:@selector(refreshTrainingWindow:)
                                                               userInfo:nil
                                                                repeats:YES];
}

- (void)createTrainingWindow {
    NSRect frame = NSMakeRect(0, 0, 900, 660);
    NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;
    self.trainingWindow = [[NSWindow alloc] initWithContentRect:frame
                                                      styleMask:style
                                                        backing:NSBackingStoreBuffered
                                                          defer:NO];
    self.trainingWindow.title = @"spec3 correction training";
    self.trainingWindow.releasedWhenClosed = NO;
    self.trainingWindow.delegate = self;
    self.trainingWindow.backgroundColor = [self windowBackgroundColor];
    self.trainingWindow.minSize = NSMakeSize(900, 660);
    [self.trainingWindow center];

    NSView *content = self.trainingWindow.contentView;

    self.trainingTitleLabel = [NSTextField labelWithString:@"Personal AI Training"];
    self.trainingTitleLabel.frame = NSMakeRect(34, 608, 430, 34);
    self.trainingTitleLabel.font = [self uiFontOfSize:26 weight:NSFontWeightBold];
    self.trainingTitleLabel.autoresizingMask = NSViewMinYMargin;
    [content addSubview:self.trainingTitleLabel];

    self.trainingModelLabel = [NSTextField labelWithString:@"Model status"];
    self.trainingModelLabel.frame = NSMakeRect(556, 615, 304, 22);
    self.trainingModelLabel.font = [self uiFontOfSize:13 weight:NSFontWeightSemibold];
    self.trainingModelLabel.textColor = [NSColor colorWithCalibratedRed:0.36 green:0.70 blue:1.00 alpha:1.0];
    self.trainingModelLabel.alignment = NSTextAlignmentRight;
    self.trainingModelLabel.autoresizingMask = NSViewMinXMargin | NSViewMinYMargin;
    [content addSubview:self.trainingModelLabel];

    self.trainingHintLabel = [NSTextField labelWithString:@""];
    self.trainingHintLabel.frame = NSMakeRect(36, 578, 824, 22);
    self.trainingHintLabel.font = [self uiFontOfSize:12 weight:NSFontWeightRegular];
    self.trainingHintLabel.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
    [content addSubview:self.trainingHintLabel];

    self.trainingQualityLabel = [self makeTrainingInfoLabel:@"" frame:NSMakeRect(36, 528, 260, 42)];
    self.trainingPoolLabel = [self makeTrainingInfoLabel:@"" frame:NSMakeRect(320, 528, 260, 42)];
    self.trainingRecordingLabel = [self makeTrainingInfoLabel:@"" frame:NSMakeRect(604, 528, 256, 42)];
    self.trainingQualityLabel.autoresizingMask = NSViewMinYMargin;
    self.trainingPoolLabel.autoresizingMask = NSViewMinYMargin;
    self.trainingRecordingLabel.autoresizingMask = NSViewMinXMargin | NSViewMinYMargin;
    [content addSubview:self.trainingQualityLabel];
    [content addSubview:self.trainingPoolLabel];
    [content addSubview:self.trainingRecordingLabel];

    self.trainingReadLabel = [self makeTrainingLabel:@"READ 0/300" y:452];
    self.trainingReadGuideLabel = [self makeTrainingGuideLabel:@"" y:426];
    self.trainingReadProgress = [self makeTrainingProgressAtY:404];
    self.trainingLiveLabel = [self makeTrainingLabel:@"LIVE 0/300" y:352];
    self.trainingLiveGuideLabel = [self makeTrainingGuideLabel:@"" y:326];
    self.trainingLiveProgress = [self makeTrainingProgressAtY:304];
    self.trainingLookLabel = [self makeTrainingLabel:@"LOOK 0/300" y:252];
    self.trainingLookGuideLabel = [self makeTrainingGuideLabel:@"" y:226];
    self.trainingLookProgress = [self makeTrainingProgressAtY:204];
    [content addSubview:self.trainingReadLabel];
    [content addSubview:self.trainingReadGuideLabel];
    [content addSubview:self.trainingReadProgress];
    [content addSubview:self.trainingLiveLabel];
    [content addSubview:self.trainingLiveGuideLabel];
    [content addSubview:self.trainingLiveProgress];
    [content addSubview:self.trainingLookLabel];
    [content addSubview:self.trainingLookGuideLabel];
    [content addSubview:self.trainingLookProgress];

    self.trainingReadButton = [self makeTrainingButton:@"Record READ" frame:NSMakeRect(664, 398, 196, 44) action:@selector(recordReadSamples:)];
    self.trainingLiveButton = [self makeTrainingButton:@"Record LIVE" frame:NSMakeRect(664, 298, 196, 44) action:@selector(recordLiveSamples:)];
    self.trainingLookButton = [self makeTrainingButton:@"Record LOOK" frame:NSMakeRect(664, 198, 196, 44) action:@selector(recordGlanceSamples:)];
    [content addSubview:self.trainingReadButton];
    [content addSubview:self.trainingLiveButton];
    [content addSubview:self.trainingLookButton];

    self.trainingNextLabel = [self makeTrainingInfoLabel:@"" frame:NSMakeRect(36, 132, 396, 54)];
    [content addSubview:self.trainingNextLabel];

    self.trainingNeedLabel = [NSTextField labelWithString:@"Need at least 20 READ and 20 non-read samples. Recommended: 300 each."];
    self.trainingNeedLabel.frame = NSMakeRect(456, 132, 404, 54);
    self.trainingNeedLabel.font = [self uiFontOfSize:13 weight:NSFontWeightRegular];
    self.trainingNeedLabel.textColor = [NSColor colorWithCalibratedRed:0.82 green:0.78 blue:0.72 alpha:1.0];
    self.trainingNeedLabel.maximumNumberOfLines = 3;
    self.trainingNeedLabel.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
    [content addSubview:self.trainingNeedLabel];

    self.trainingStatusLabel = [NSTextField labelWithString:@"Waiting for training state..."];
    self.trainingStatusLabel.frame = NSMakeRect(38, 82, 822, 34);
    self.trainingStatusLabel.font = [self uiFontOfSize:14 weight:NSFontWeightSemibold];
    self.trainingStatusLabel.textColor = [NSColor colorWithCalibratedRed:0.98 green:0.62 blue:0.28 alpha:1.0];
    self.trainingStatusLabel.lineBreakMode = NSLineBreakByTruncatingTail;
    self.trainingStatusLabel.maximumNumberOfLines = 2;
    self.trainingStatusLabel.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
    [content addSubview:self.trainingStatusLabel];

    self.trainingStopButton = [self makeTrainingButton:@"Stop Recording" frame:NSMakeRect(38, 26, 156, 42) action:@selector(stopRecordingSamples:)];
    self.trainingResetButton = [self makeTrainingButton:@"Reset Samples" frame:NSMakeRect(212, 26, 156, 42) action:@selector(resetTrainingSamples:)];
    self.trainingFolderButton = [self makeTrainingButton:@"Data Folder" frame:NSMakeRect(386, 26, 156, 42) action:@selector(openTrainingDataFolder:)];
    self.trainingTrainButton = [self makeTrainingButton:@"Train Model" frame:NSMakeRect(676, 26, 184, 42) action:@selector(trainPersonalAI:)];
    [content addSubview:self.trainingStopButton];
    [content addSubview:self.trainingResetButton];
    [content addSubview:self.trainingFolderButton];
    [content addSubview:self.trainingTrainButton];
    [self applyTrainingWindowStyle];
}

- (NSTextField *)makeTrainingLabel:(NSString *)text y:(CGFloat)y {
    NSTextField *label = [NSTextField labelWithString:text];
    label.frame = NSMakeRect(38, y, 420, 26);
    label.font = [self uiFontOfSize:17 weight:NSFontWeightSemibold];
    label.textColor = [NSColor colorWithCalibratedRed:0.93 green:0.91 blue:0.87 alpha:1.0];
    label.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
    return label;
}

- (NSTextField *)makeTrainingGuideLabel:(NSString *)text y:(CGFloat)y {
    NSTextField *label = [NSTextField labelWithString:text];
    label.frame = NSMakeRect(40, y, 590, 20);
    label.font = [self uiFontOfSize:12 weight:NSFontWeightRegular];
    label.textColor = [self secondaryTextColor];
    label.lineBreakMode = NSLineBreakByTruncatingTail;
    label.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
    return label;
}

- (NSProgressIndicator *)makeTrainingProgressAtY:(CGFloat)y {
    NSProgressIndicator *progress = [[NSProgressIndicator alloc] initWithFrame:NSMakeRect(40, y, 590, 14)];
    progress.indeterminate = NO;
    progress.minValue = 0.0;
    progress.maxValue = 100.0;
    progress.doubleValue = 0.0;
    progress.controlSize = NSControlSizeRegular;
    progress.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
    return progress;
}

- (NSTextField *)makeTrainingInfoLabel:(NSString *)text frame:(NSRect)frame {
    NSTextField *label = [NSTextField labelWithString:text];
    label.frame = frame;
    label.font = [self uiFontOfSize:12 weight:NSFontWeightSemibold];
    label.textColor = [self primaryTextColor];
    label.maximumNumberOfLines = 2;
    label.lineBreakMode = NSLineBreakByTruncatingTail;
    label.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
    return label;
}

- (NSButton *)makeTrainingButton:(NSString *)title frame:(NSRect)frame action:(SEL)action {
    NSButton *button = [NSButton buttonWithTitle:title target:self action:action];
    button.frame = frame;
    button.bezelStyle = NSBezelStyleRounded;
    button.autoresizingMask = NSViewMinXMargin | NSViewMinYMargin;
    [self styleButton:button title:title emphasized:NO];
    return button;
}

- (void)applyTrainingWindowStyle {
    if (!self.trainingWindow) {
        return;
    }
    [self applyAppearanceToWindow:self.trainingWindow];
    self.trainingWindow.backgroundColor = [self windowBackgroundColor];
    self.trainingTitleLabel.stringValue = [self textEN:@"Personal AI Training" ru:@"Обучение Personal AI"];
    self.trainingTitleLabel.font = [self uiFontOfSize:26 weight:NSFontWeightBold];
    self.trainingTitleLabel.textColor = [self primaryTextColor];
    self.trainingModelLabel.font = [self uiFontOfSize:13 weight:NSFontWeightSemibold];
    self.trainingModelLabel.textColor = [self accentColor];
    self.trainingHintLabel.stringValue = [self textEN:@"Record clean examples, train, then keep adding short fresh sessions when READ/LIVE feels wrong." ru:@"Записывай чистые примеры, обучай, потом добавляй короткие свежие сессии, если READ/LIVE путается."];
    self.trainingHintLabel.font = [self uiFontOfSize:12 weight:NSFontWeightRegular];
    self.trainingHintLabel.textColor = [self secondaryTextColor];
    self.trainingQualityLabel.font = [self uiFontOfSize:12 weight:NSFontWeightSemibold];
    self.trainingPoolLabel.font = [self uiFontOfSize:12 weight:NSFontWeightSemibold];
    self.trainingRecordingLabel.font = [self uiFontOfSize:12 weight:NSFontWeightSemibold];
    self.trainingNextLabel.font = [self uiFontOfSize:13 weight:NSFontWeightSemibold];
    self.trainingQualityLabel.textColor = [self primaryTextColor];
    self.trainingPoolLabel.textColor = [self primaryTextColor];
    self.trainingRecordingLabel.textColor = [self primaryTextColor];
    self.trainingNextLabel.textColor = [self primaryTextColor];
    self.trainingReadLabel.font = [self uiFontOfSize:17 weight:NSFontWeightSemibold];
    self.trainingLiveLabel.font = [self uiFontOfSize:17 weight:NSFontWeightSemibold];
    self.trainingLookLabel.font = [self uiFontOfSize:17 weight:NSFontWeightSemibold];
    self.trainingReadLabel.textColor = [self primaryTextColor];
    self.trainingLiveLabel.textColor = [self primaryTextColor];
    self.trainingLookLabel.textColor = [self primaryTextColor];
    self.trainingReadGuideLabel.stringValue = [self textEN:@"Read centered text for 20-40 seconds. Keep your head calm; do not look aside." ru:@"Читай текст по центру 20-40 секунд. Голову держи спокойно, в сторону не смотри."];
    self.trainingLiveGuideLabel.stringValue = [self textEN:@"Look naturally at the camera or one point. Do not read lines of text." ru:@"Смотри естественно в камеру или одну точку. Не читай строки текста."];
    self.trainingLookGuideLabel.stringValue = [self textEN:@"Make normal side glances: left, right, up, down, then back." ru:@"Делай обычные взгляды в сторону: влево, вправо, вверх, вниз и обратно."];
    self.trainingReadGuideLabel.font = [self uiFontOfSize:12 weight:NSFontWeightRegular];
    self.trainingLiveGuideLabel.font = [self uiFontOfSize:12 weight:NSFontWeightRegular];
    self.trainingLookGuideLabel.font = [self uiFontOfSize:12 weight:NSFontWeightRegular];
    self.trainingReadGuideLabel.textColor = [self secondaryTextColor];
    self.trainingLiveGuideLabel.textColor = [self secondaryTextColor];
    self.trainingLookGuideLabel.textColor = [self secondaryTextColor];
    self.trainingNeedLabel.font = [self uiFontOfSize:13 weight:NSFontWeightRegular];
    self.trainingStatusLabel.font = [self uiFontOfSize:14 weight:NSFontWeightSemibold];
    self.trainingStatusLabel.textColor = [self accentColor];
    [self styleButton:self.trainingStopButton title:[self textEN:@"Stop Recording" ru:@"Стоп"] emphasized:NO];
    [self styleButton:self.trainingResetButton title:[self textEN:@"Reset Samples" ru:@"Обнулить"] emphasized:NO];
    [self styleButton:self.trainingFolderButton title:[self textEN:@"Data Folder" ru:@"Папка данных"] emphasized:NO];
    [self styleButton:self.trainingTrainButton title:[self textEN:@"Train Model" ru:@"Обучить"] emphasized:YES];
}

- (void)refreshTrainingWindowSoon {
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.25 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
        [self refreshTrainingWindow:nil];
    });
}

- (NSDictionary *)readTrainingState {
    NSData *data = [NSData dataWithContentsOfFile:self.trainingStatePath];
    if (!data) {
        return @{};
    }
    id json = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    if (![json isKindOfClass:[NSDictionary class]]) {
        return @{};
    }
    return (NSDictionary *)json;
}

- (void)refreshTrainingWindow:(id)sender {
    if (!self.trainingWindow) {
        return;
    }
    [self applyTrainingWindowStyle];
    NSDictionary *state = [self readTrainingState];
    NSDictionary *samples = [state[@"samples"] isKindOfClass:[NSDictionary class]] ? state[@"samples"] : @{};
    NSDictionary *pool = [state[@"training_pool"] isKindOfClass:[NSDictionary class]] ? state[@"training_pool"] : samples;
    NSInteger target = [state[@"target_per_label"] integerValue] > 0 ? [state[@"target_per_label"] integerValue] : 300;
    NSInteger read = [samples[@"read"] integerValue];
    NSInteger live = [samples[@"live"] integerValue];
    NSInteger look = [samples[@"glance"] integerValue];
    NSInteger poolRead = [pool[@"read"] integerValue];
    NSInteger poolLive = [pool[@"live"] integerValue];
    NSInteger poolLook = [pool[@"glance"] integerValue];
    NSInteger poolTotal = poolRead + poolLive + poolLook;
    NSInteger poolNonRead = poolLive + poolLook;
    NSInteger needRead = MAX(0, 20 - poolRead);
    NSInteger needNonRead = MAX(0, 20 - poolNonRead);
    NSString *recording = [state[@"recording_label"] isKindOfClass:[NSString class]] ? state[@"recording_label"] : @"";
    NSString *status = [state[@"last_status"] isKindOfClass:[NSString class]] ? state[@"last_status"] : @"Open the camera, then choose a recording mode.";
    BOOL hasModel = [state[@"has_model"] boolValue];
    BOOL canTrain = [state[@"can_train"] boolValue];
    NSInteger modelSamples = [state[@"model_samples"] integerValue];
    double modelAccuracy = [state[@"model_accuracy"] doubleValue];
    double modelThreshold = [state[@"model_threshold"] doubleValue];
    NSInteger historyFiles = [state[@"history_files"] integerValue];
    double warmupSeconds = [state[@"recording_warmup_seconds"] doubleValue];

    self.trainingReadLabel.stringValue = [NSString stringWithFormat:@"%@  %ld / %ld", [self textEN:@"READ now" ru:@"ЧТЕНИЕ сейчас"], (long)read, (long)target];
    self.trainingLiveLabel.stringValue = [NSString stringWithFormat:@"%@  %ld / %ld", [self textEN:@"LIVE now" ru:@"ЖИВОЙ сейчас"], (long)live, (long)target];
    self.trainingLookLabel.stringValue = [NSString stringWithFormat:@"%@  %ld / %ld", [self textEN:@"LOOK now" ru:@"ВЗГЛЯД сейчас"], (long)look, (long)target];
    self.trainingReadProgress.doubleValue = MIN(100.0, (double)read * 100.0 / MAX(target, 1));
    self.trainingLiveProgress.doubleValue = MIN(100.0, (double)live * 100.0 / MAX(target, 1));
    self.trainingLookProgress.doubleValue = MIN(100.0, (double)look * 100.0 / MAX(target, 1));

    NSString *recordingTitle = [self textEN:@"ready" ru:@"готово"];
    if ([recording isEqualToString:@"read"]) {
        recordingTitle = [self textEN:@"recording READ" ru:@"пишу ЧТЕНИЕ"];
    } else if ([recording isEqualToString:@"live"]) {
        recordingTitle = [self textEN:@"recording LIVE" ru:@"пишу ЖИВОЙ"];
    } else if ([recording isEqualToString:@"glance"]) {
        recordingTitle = [self textEN:@"recording LOOK" ru:@"пишу ВЗГЛЯД"];
    }

    if (hasModel && modelSamples > 0) {
        if ([self isRussian]) {
            self.trainingQualityLabel.stringValue = [NSString stringWithFormat:@"Качество модели\n%.0f%% • %ld примеров • порог %.2f", modelAccuracy * 100.0, (long)modelSamples, modelThreshold];
        } else {
            self.trainingQualityLabel.stringValue = [NSString stringWithFormat:@"Model quality\n%.0f%% • %ld samples • %.2f gate", modelAccuracy * 100.0, (long)modelSamples, modelThreshold];
        }
    } else {
        self.trainingQualityLabel.stringValue = [self textEN:@"Model quality\nnot trained yet" ru:@"Качество модели\nещё не обучена"];
    }
    if ([self isRussian]) {
        self.trainingPoolLabel.stringValue = [NSString stringWithFormat:@"Сохранённая база\n%ld всего • %ld сессий", (long)poolTotal, (long)historyFiles];
        self.trainingRecordingLabel.stringValue = [NSString stringWithFormat:@"Запись\n%@ • старт %.1fс не берём", recordingTitle, warmupSeconds > 0.0 ? warmupSeconds : 0.6];
    } else {
        self.trainingPoolLabel.stringValue = [NSString stringWithFormat:@"Saved dataset\n%ld total • %ld sessions", (long)poolTotal, (long)historyFiles];
        self.trainingRecordingLabel.stringValue = [NSString stringWithFormat:@"Recording\n%@ • %.1fs warmup skipped", recordingTitle, warmupSeconds > 0.0 ? warmupSeconds : 0.6];
    }

    if (needRead == 0 && needNonRead == 0) {
        if ([self isRussian]) {
            self.trainingNeedLabel.stringValue = [NSString stringWithFormat:@"Данных достаточно: %ld всего.\nДальше добавляй только свежие примеры, если READ/LIVE путается.", (long)poolTotal];
        } else {
            self.trainingNeedLabel.stringValue = [NSString stringWithFormat:@"Enough data: %ld total.\nAdd fresh samples only when READ/LIVE feels wrong.", (long)poolTotal];
        }
        self.trainingNeedLabel.textColor = [NSColor colorWithCalibratedRed:0.52 green:0.86 blue:0.56 alpha:1.0];
    } else {
        if ([self isRussian]) {
            self.trainingNeedLabel.stringValue = [NSString stringWithFormat:@"Минимум: ещё ЧТЕНИЕ %ld, ЖИВОЙ/ВЗГЛЯД %ld.\nВ истории уже %ld примеров.", (long)needRead, (long)needNonRead, (long)poolTotal];
        } else {
            self.trainingNeedLabel.stringValue = [NSString stringWithFormat:@"Minimum: %ld more READ, %ld more LIVE/LOOK.\nHistory already has %ld samples.", (long)needRead, (long)needNonRead, (long)poolTotal];
        }
        self.trainingNeedLabel.textColor = [self secondaryTextColor];
    }

    if (recording.length > 0) {
        self.trainingNextLabel.stringValue = [self textEN:@"Now\nKeep this state clean; press Stop before switching." ru:@"Сейчас\nДержи пример чистым; перед сменой нажми Стоп."];
        self.trainingNextLabel.textColor = [self accentColor];
    } else if (needRead > 0 || poolRead < target) {
        self.trainingNextLabel.stringValue = [self textEN:@"Next step\nRecord READ with text centered on screen." ru:@"Следующий шаг\nЗапиши ЧТЕНИЕ с текстом по центру экрана."];
        self.trainingNextLabel.textColor = [self primaryTextColor];
    } else if (poolLive < target) {
        self.trainingNextLabel.stringValue = [self textEN:@"Next step\nRecord LIVE: natural gaze, no reading." ru:@"Следующий шаг\nЗапиши ЖИВОЙ: естественный взгляд без чтения."];
        self.trainingNextLabel.textColor = [self primaryTextColor];
    } else if (poolLook < target) {
        self.trainingNextLabel.stringValue = [self textEN:@"Next step\nRecord LOOK: short side glances." ru:@"Следующий шаг\nЗапиши ВЗГЛЯД: короткие взгляды в стороны."];
        self.trainingNextLabel.textColor = [self primaryTextColor];
    } else if (canTrain && (!hasModel || (read + live + look) > 30)) {
        self.trainingNextLabel.stringValue = [self textEN:@"Next step\nTrain, then test READ/LIVE switching." ru:@"Следующий шаг\nОбучи и проверь переключение READ/LIVE."];
        self.trainingNextLabel.textColor = [self accentColor];
    } else {
        self.trainingNextLabel.stringValue = [self textEN:@"Next step\nIf READ is wrong, add a short fresh session." ru:@"Следующий шаг\nЕсли READ путается, добавь короткую свежую сессию."];
        self.trainingNextLabel.textColor = [self primaryTextColor];
    }

    if (hasModel && modelSamples > 0) {
        if ([self isRussian]) {
            self.trainingModelLabel.stringValue = [NSString stringWithFormat:@"модель готова • %ld • %.0f%%", (long)modelSamples, modelAccuracy * 100.0];
        } else {
            self.trainingModelLabel.stringValue = [NSString stringWithFormat:@"model ON • %ld • %.0f%%", (long)modelSamples, modelAccuracy * 100.0];
        }
    } else {
        self.trainingModelLabel.stringValue = hasModel ? [self textEN:@"model ON" ru:@"модель готова"] : [self textEN:@"model not trained" ru:@"модель не обучена"];
    }
    if ([self isRussian] && [status containsString:@"Personal AI trained"]) {
        self.trainingStatusLabel.stringValue = [NSString stringWithFormat:@"Personal AI обучена: %ld примеров, точность %.0f%%. Новые записи очищены, история сохранена.", (long)modelSamples, modelAccuracy * 100.0];
    } else if ([self isRussian] && [status isEqualToString:@"Personal AI ready"]) {
        self.trainingStatusLabel.stringValue = @"Personal AI готова.";
    } else {
        self.trainingStatusLabel.stringValue = status;
    }
    self.trainingTrainButton.enabled = YES;
    [self styleButton:self.trainingReadButton title:([recording isEqualToString:@"read"] ? [self textEN:@"Recording READ" ru:@"Пишу ЧТЕНИЕ"] : [self textEN:@"Record READ" ru:@"Писать ЧТЕНИЕ"]) emphasized:[recording isEqualToString:@"read"]];
    [self styleButton:self.trainingLiveButton title:([recording isEqualToString:@"live"] ? [self textEN:@"Recording LIVE" ru:@"Пишу ЖИВОЙ"] : [self textEN:@"Record LIVE" ru:@"Писать ЖИВОЙ"]) emphasized:[recording isEqualToString:@"live"]];
    [self styleButton:self.trainingLookButton title:([recording isEqualToString:@"glance"] ? [self textEN:@"Recording LOOK" ru:@"Пишу ВЗГЛЯД"] : [self textEN:@"Record LOOK" ru:@"Писать ВЗГЛЯД"]) emphasized:[recording isEqualToString:@"glance"]];
    [self styleButton:self.trainingTrainButton title:(canTrain ? [self textEN:@"Train Model" ru:@"Обучить"] : [self textEN:@"Train Anyway" ru:@"Пробовать обучить"]) emphasized:YES];
}

- (void)showSettingsWindow:(id)sender {
    if (!self.settingsWindow) {
        [self createSettingsWindow];
    }
    [self refreshSettingsWindow];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    [NSApp activateIgnoringOtherApps:YES];
    [self.settingsWindow makeKeyAndOrderFront:nil];
}

- (void)createSettingsWindow {
    NSRect frame = NSMakeRect(0, 0, 560, 470);
    NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable;
    self.settingsWindow = [[NSWindow alloc] initWithContentRect:frame
                                                      styleMask:style
                                                        backing:NSBackingStoreBuffered
                                                          defer:NO];
    self.settingsWindow.title = @"spec3 correction settings";
    self.settingsWindow.releasedWhenClosed = NO;
    self.settingsWindow.delegate = self;
    [self.settingsWindow center];

    NSView *content = self.settingsWindow.contentView;
    self.settingsTitleLabel = [NSTextField labelWithString:@""];
    self.settingsTitleLabel.frame = NSMakeRect(34, 402, 410, 34);
    self.settingsTitleLabel.font = [self uiFontOfSize:26 weight:NSFontWeightBold];
    [content addSubview:self.settingsTitleLabel];

    self.settingsThemeLabel = [NSTextField labelWithString:@""];
    self.settingsThemeLabel.frame = NSMakeRect(40, 338, 160, 24);
    self.settingsThemeLabel.font = [self uiFontOfSize:15 weight:NSFontWeightSemibold];
    [content addSubview:self.settingsThemeLabel];

    self.themePopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(228, 333, 260, 32)];
    [self.themePopup addItemWithTitle:@"Dark"];
    [self.themePopup addItemWithTitle:@"Light"];
    [content addSubview:self.themePopup];

    self.settingsStyleLabel = [NSTextField labelWithString:@""];
    self.settingsStyleLabel.frame = NSMakeRect(40, 282, 160, 24);
    self.settingsStyleLabel.font = [self uiFontOfSize:15 weight:NSFontWeightSemibold];
    [content addSubview:self.settingsStyleLabel];

    self.stylePopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(228, 277, 260, 32)];
    [self.stylePopup addItemWithTitle:@"Orange"];
    [self.stylePopup addItemWithTitle:@"Blue"];
    [self.stylePopup addItemWithTitle:@"Mint"];
    [self.stylePopup addItemWithTitle:@"Violet"];
    [self.stylePopup addItemWithTitle:@"Graphite"];
    [content addSubview:self.stylePopup];

    self.settingsFontLabel = [NSTextField labelWithString:@""];
    self.settingsFontLabel.frame = NSMakeRect(40, 226, 160, 24);
    self.settingsFontLabel.font = [self uiFontOfSize:15 weight:NSFontWeightSemibold];
    [content addSubview:self.settingsFontLabel];

    self.fontPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(228, 221, 260, 32)];
    [self.fontPopup addItemWithTitle:@"Rounded"];
    [self.fontPopup addItemWithTitle:@"System"];
    [self.fontPopup addItemWithTitle:@"Compact"];
    [self.fontPopup addItemWithTitle:@"Mono"];
    [self.fontPopup addItemWithTitle:@"Serif"];
    [content addSubview:self.fontPopup];

    self.settingsLanguageLabel = [NSTextField labelWithString:@""];
    self.settingsLanguageLabel.frame = NSMakeRect(40, 170, 160, 24);
    self.settingsLanguageLabel.font = [self uiFontOfSize:15 weight:NSFontWeightSemibold];
    [content addSubview:self.settingsLanguageLabel];

    self.languagePopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(228, 165, 260, 32)];
    [self.languagePopup addItemWithTitle:@"English"];
    [self.languagePopup addItemWithTitle:@"Русский"];
    [content addSubview:self.languagePopup];

    self.settingsStatusLabel = [NSTextField labelWithString:@""];
    self.settingsStatusLabel.frame = NSMakeRect(40, 102, 448, 42);
    self.settingsStatusLabel.font = [self uiFontOfSize:13 weight:NSFontWeightRegular];
    self.settingsStatusLabel.maximumNumberOfLines = 2;
    [content addSubview:self.settingsStatusLabel];

    self.settingsSaveButton = [NSButton buttonWithTitle:@"Save" target:self action:@selector(saveSettings:)];
    self.settingsSaveButton.frame = NSMakeRect(370, 38, 118, 38);
    self.settingsSaveButton.bezelStyle = NSBezelStyleRounded;
    [content addSubview:self.settingsSaveButton];
}

- (void)refreshSettingsWindow {
    if (!self.settingsWindow) {
        return;
    }
    NSDictionary *prefs = [self readPreferences];
    NSString *theme = [prefs[@"theme"] isKindOfClass:[NSString class]] ? prefs[@"theme"] : @"dark";
    NSString *language = [prefs[@"language"] isKindOfClass:[NSString class]] ? prefs[@"language"] : @"en";
    NSString *style = [self selectedStyleKey];
    NSString *font = [self selectedFontKey];
    [self.themePopup selectItemWithTitle:[theme isEqualToString:@"light"] ? @"Light" : @"Dark"];
    [self.languagePopup selectItemWithTitle:[language isEqualToString:@"ru"] ? @"Русский" : @"English"];
    [self.stylePopup selectItemWithTitle:[style capitalizedString]];
    if ([font isEqualToString:@"mono"]) {
        [self.fontPopup selectItemWithTitle:@"Mono"];
    } else if ([font isEqualToString:@"serif"]) {
        [self.fontPopup selectItemWithTitle:@"Serif"];
    } else if ([font isEqualToString:@"compact"]) {
        [self.fontPopup selectItemWithTitle:@"Compact"];
    } else if ([font isEqualToString:@"system"]) {
        [self.fontPopup selectItemWithTitle:@"System"];
    } else {
        [self.fontPopup selectItemWithTitle:@"Rounded"];
    }
    [self applyAppearanceToWindow:self.settingsWindow];
    self.settingsWindow.backgroundColor = [self windowBackgroundColor];
    self.themePopup.appearance = [self preferredAppearance];
    self.stylePopup.appearance = [self preferredAppearance];
    self.fontPopup.appearance = [self preferredAppearance];
    self.languagePopup.appearance = [self preferredAppearance];
    self.settingsTitleLabel.stringValue = [self textEN:@"Settings" ru:@"Настройки"];
    self.settingsThemeLabel.stringValue = [self textEN:@"Theme" ru:@"Тема"];
    self.settingsStyleLabel.stringValue = [self textEN:@"Color style" ru:@"Стиль цвета"];
    self.settingsFontLabel.stringValue = [self textEN:@"Font" ru:@"Шрифт"];
    self.settingsLanguageLabel.stringValue = [self textEN:@"Language" ru:@"Язык"];
    self.settingsStatusLabel.stringValue = [self textEN:@"Style and font update the main panel, training and settings windows." ru:@"Стиль и шрифт меняют главную панель, обучение и настройки."];
    self.settingsTitleLabel.font = [self uiFontOfSize:26 weight:NSFontWeightBold];
    self.settingsThemeLabel.font = [self uiFontOfSize:15 weight:NSFontWeightSemibold];
    self.settingsStyleLabel.font = [self uiFontOfSize:15 weight:NSFontWeightSemibold];
    self.settingsFontLabel.font = [self uiFontOfSize:15 weight:NSFontWeightSemibold];
    self.settingsLanguageLabel.font = [self uiFontOfSize:15 weight:NSFontWeightSemibold];
    self.settingsStatusLabel.font = [self uiFontOfSize:13 weight:NSFontWeightRegular];
    self.settingsTitleLabel.textColor = [self primaryTextColor];
    self.settingsThemeLabel.textColor = [self primaryTextColor];
    self.settingsStyleLabel.textColor = [self primaryTextColor];
    self.settingsFontLabel.textColor = [self primaryTextColor];
    self.settingsLanguageLabel.textColor = [self primaryTextColor];
    self.settingsStatusLabel.textColor = [self secondaryTextColor];
    [self styleButton:self.settingsSaveButton title:[self textEN:@"Save" ru:@"Сохранить"] emphasized:YES];
}

- (void)saveSettings:(id)sender {
    NSString *theme = [[self.themePopup titleOfSelectedItem] isEqualToString:@"Light"] ? @"light" : @"dark";
    NSString *language = [[self.languagePopup titleOfSelectedItem] isEqualToString:@"Русский"] ? @"ru" : @"en";
    NSString *style = [[[self.stylePopup titleOfSelectedItem] lowercaseString] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSString *font = [[[self.fontPopup titleOfSelectedItem] lowercaseString] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    [self writePreferencesWithTheme:theme language:language style:style font:font];
    [self refreshSettingsWindow];
    [self applyTrainingWindowStyle];
    [self refreshTrainingWindow:nil];
    self.settingsStatusLabel.stringValue = [self textEN:@"Saved." ru:@"Сохранено."];
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
    if (notification.object == self.trainingWindow) {
        [self.trainingRefreshTimer invalidate];
        self.trainingRefreshTimer = nil;
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

- (void)processNativeRequest:(NSTimer *)timer {
    if (!self.nativeRequestPath) {
        return;
    }
    NSDictionary *attrs = [[NSFileManager defaultManager] attributesOfItemAtPath:self.nativeRequestPath error:nil];
    NSDate *date = attrs[NSFileModificationDate];
    if (!date) {
        return;
    }
    NSTimeInterval mtime = [date timeIntervalSince1970];
    if (mtime <= self.lastNativeRequestMTime) {
        return;
    }
    self.lastNativeRequestMTime = mtime;

    NSString *payload = [NSString stringWithContentsOfFile:self.nativeRequestPath encoding:NSUTF8StringEncoding error:nil];
    NSString *firstLine = [[payload componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]] firstObject];
    NSString *command = [[firstLine stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] lowercaseString];
    if (command.length == 0) {
        return;
    }

    [self logLine:[NSString stringWithFormat:@"Native request: %@", command]];
    [[NSFileManager defaultManager] removeItemAtPath:self.nativeRequestPath error:nil];
    if ([command isEqualToString:@"training"]) {
        [self showTrainingWindow:nil];
    } else if ([command isEqualToString:@"settings"]) {
        [self showSettingsWindow:nil];
    } else if ([command isEqualToString:@"logs"]) {
        [self showLogWindow:nil];
    } else if ([command isEqualToString:@"show"]) {
        [self showWindow:nil];
    } else if ([command isEqualToString:@"hide"]) {
        [self hideWindow:nil];
    } else if ([command isEqualToString:@"quit"]) {
        [NSApp terminate:nil];
    }
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

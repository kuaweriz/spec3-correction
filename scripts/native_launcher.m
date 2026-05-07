#import <AVFoundation/AVFoundation.h>
#import <Cocoa/Cocoa.h>

static void ShowAlert(NSString *title, NSString *message) {
    dispatch_async(dispatch_get_main_queue(), ^{
        NSAlert *alert = [[NSAlert alloc] init];
        [alert setMessageText:title];
        [alert setInformativeText:message];
        [alert addButtonWithTitle:@"OK"];
        [alert runModal];
        [NSApp terminate:nil];
    });
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        [NSApplication sharedApplication];

        NSBundle *bundle = [NSBundle mainBundle];
        NSString *resourcePath = [bundle resourcePath];
        NSString *projectPath = [resourcePath stringByAppendingPathComponent:@"project"];
        NSString *pythonPath = [projectPath stringByAppendingPathComponent:@".venv/bin/python"];
        NSString *scriptPath = [projectPath stringByAppendingPathComponent:@"bin_single_window.py"];

        NSString *home = NSHomeDirectory();
        NSString *logDir = [home stringByAppendingPathComponent:@"Library/Logs"];
        NSString *logPath = [logDir stringByAppendingPathComponent:@"Gaze Correction Camera.log"];
        NSString *cacheDir = [[home stringByAppendingPathComponent:@"Library/Caches"] stringByAppendingPathComponent:@"Gaze Correction Camera/matplotlib"];

        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:logDir withIntermediateDirectories:YES attributes:nil error:nil];
        [fm createDirectoryAtPath:cacheDir withIntermediateDirectories:YES attributes:nil error:nil];

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
            ShowAlert(
                @"Camera access is blocked",
                @"Open System Settings > Privacy & Security > Camera and allow Gaze Correction Camera, then launch it again."
            );
            [NSApp run];
            return 1;
        }

        NSFileHandle *logHandle = [NSFileHandle fileHandleForWritingAtPath:logPath];
        if (!logHandle) {
            [@"" writeToFile:logPath atomically:YES encoding:NSUTF8StringEncoding error:nil];
            logHandle = [NSFileHandle fileHandleForWritingAtPath:logPath];
        }
        [logHandle seekToEndOfFile];
        NSString *startLine = [NSString stringWithFormat:@"\n--- Gaze Correction Camera native start: %@ ---\n", [NSDate date]];
        [logHandle writeData:[startLine dataUsingEncoding:NSUTF8StringEncoding]];

        NSTask *task = [[NSTask alloc] init];
        [task setCurrentDirectoryPath:projectPath];
        [task setLaunchPath:@"/usr/bin/arch"];
        [task setArguments:@[@"-arm64", pythonPath, scriptPath, @"--backend", @"mediapipe"]];

        NSMutableDictionary *env = [[[NSProcessInfo processInfo] environment] mutableCopy];
        env[@"PYTHONUNBUFFERED"] = @"1";
        env[@"PYTHONNOUSERSITE"] = @"1";
        env[@"MPLCONFIGDIR"] = cacheDir;
        [task setEnvironment:env];
        [task setStandardOutput:logHandle];
        [task setStandardError:logHandle];

        @try {
            [task launch];
            [task waitUntilExit];
        } @catch (NSException *exception) {
            NSString *message = [NSString stringWithFormat:@"Could not launch Python runtime: %@", [exception reason]];
            [logHandle writeData:[message dataUsingEncoding:NSUTF8StringEncoding]];
            ShowAlert(@"Gaze Correction Camera could not start", message);
            [NSApp run];
            return 1;
        }

        int exitCode = [task terminationStatus];
        if (exitCode != 0) {
            ShowAlert(
                @"Gaze Correction Camera could not start",
                @"Check the log at ~/Library/Logs/Gaze Correction Camera.log."
            );
            [NSApp run];
            return exitCode;
        }

        return 0;
    }
}

#import <Cocoa/Cocoa.h>

static void RunTask(NSString *launchPath, NSArray<NSString *> *arguments) {
    NSTask *task = [[NSTask alloc] init];
    [task setLaunchPath:launchPath];
    [task setArguments:arguments];
    @try {
        [task launch];
        [task waitUntilExit];
    } @catch (NSException *exception) {
    }
}

static void ShowDone(void) {
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setMessageText:@"SpecTree stopped"];
    [alert setInformativeText:@"Camera processes were closed."];
    [alert addButtonWithTitle:@"OK"];
    [alert runModal];
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        [NSApplication sharedApplication];
        RunTask(@"/usr/bin/pkill", @[@"-f", @"bin_single_window.py"]);
        RunTask(@"/usr/bin/pkill", @[@"-f", @"SpecTree.app/Contents/MacOS/SpecTree"]);
        RunTask(@"/usr/bin/pkill", @[@"-f", @"Gaze Correction Camera.app/Contents/MacOS/Gaze Correction Camera"]);
        ShowDone();
        return 0;
    }
}

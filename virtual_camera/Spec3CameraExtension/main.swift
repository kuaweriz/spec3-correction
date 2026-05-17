import CoreMediaIO
import Foundation

let providerSource = Spec3ProviderSource(clientQueue: nil)
CMIOExtensionProvider.startService(provider: providerSource.provider)

CFRunLoopRun()

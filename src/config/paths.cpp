#include "config/paths.h"

#include <cstdlib>
#include <filesystem>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <climits>
#else
#include <climits>
#include <unistd.h>
#endif

namespace dray::config {

namespace {

std::string env(const char* name) {
    const char* v = std::getenv(name);   // NOLINT: read-only, at startup
    return v ? std::string(v) : std::string();
}

std::filesystem::path executable_dir() {
#if defined(_WIN32)
    wchar_t buf[32768];
    const DWORD n = GetModuleFileNameW(nullptr, buf, static_cast<DWORD>(std::size(buf)));
    if (n == 0 || n >= std::size(buf)) return {};
    return std::filesystem::path(std::wstring(buf, n)).parent_path();
#elif defined(__APPLE__)
    char buf[PATH_MAX];
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) != 0) return {};
    std::error_code ec;
    const auto p = std::filesystem::weakly_canonical(buf, ec);
    return ec ? std::filesystem::path(buf).parent_path() : p.parent_path();
#else
    char buf[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return {};
    buf[n] = '\0';
    return std::filesystem::path(buf).parent_path();
#endif
}

std::filesystem::path user_dir(const std::string& config_dir) {
    if (!config_dir.empty()) return config_dir;
    const std::string over = env("DRAY_CONFIG_DIR");
    if (!over.empty()) return over;
#if defined(_WIN32)
    const std::string appdata = env("APPDATA");
    return appdata.empty() ? std::filesystem::path() : std::filesystem::path(appdata) / "dray";
#elif defined(__APPLE__)
    const std::string home = env("HOME");
    return home.empty() ? std::filesystem::path()
                        : std::filesystem::path(home) / "Library" / "Application Support" / "dray";
#else
    const std::string xdg = env("XDG_CONFIG_HOME");
    if (!xdg.empty()) return std::filesystem::path(xdg) / "dray";
    const std::string home = env("HOME");
    return home.empty() ? std::filesystem::path() : std::filesystem::path(home) / ".config" / "dray";
#endif
}

}  // namespace

Locations default_locations(const std::string& config_dir) {
    Locations l;
    const std::filesystem::path exe = executable_dir();
    if (!exe.empty()) l.shipped = exe / "config";
    l.user = user_dir(config_dir);
    return l;
}

}  // namespace dray::config

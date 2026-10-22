"""Exercise automatic function exports and runtime DLL staging on a real Windows linker."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name == 'nt', 'Requires the Windows loader and MSVC')
class WindowsKernelDllTest(unittest.TestCase):
    def test_exported_bridge_runs_only_with_staged_dll(self):
        module = Path(__file__).resolve().parents[2] / 'cmake/NinferTargets.cmake'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'CMakeLists.txt').write_text(f'''
cmake_minimum_required(VERSION 3.28)
project(kernel_dll_fixture LANGUAGES CXX)
set(NINFER_WINDOWS_KERNEL_DLL ON)
include("{module.as_posix()}")
add_library(ninfer_ggml_quants SHARED kernel.cpp)
set_target_properties(ninfer_ggml_quants PROPERTIES
  WINDOWS_EXPORT_ALL_SYMBOLS ON DEFINE_SYMBOL "" POSITION_INDEPENDENT_CODE OFF
  RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/kernels")
add_executable(consumer main.cpp)
set_target_properties(consumer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/apps")
target_link_libraries(consumer PRIVATE ninfer_ggml_quants)
ninfer_stage_runtime_dlls(consumer)
''')
            (root / 'kernel.cpp').write_text('namespace bridge { int value() { return 42; } }\n')
            (root / 'main.cpp').write_text(
                'namespace bridge { int value(); } int main() { return bridge::value() != 42; }\n')
            build = root / 'build'
            for command in (['cmake', '-S', str(root), '-B', str(build), '-G', 'Visual Studio 17 2022', '-A', 'x64'],
                            ['cmake', '--build', str(build), '--config', 'Release', '-j']):
                result = subprocess.run(command, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            folder = build / 'apps/Release'
            exe = folder / 'consumer.exe'
            dll = folder / 'ninfer_ggml_quants.dll'
            self.assertTrue(dll.is_file())
            env = dict(os.environ, PATH=os.environ['SystemRoot'] + r'\System32')
            self.assertEqual(subprocess.run([str(exe)], cwd=root, env=env, timeout=10).returncode, 0)
            dll.rename(dll.with_suffix('.hidden'))
            self.assertNotEqual(subprocess.run([str(exe)], cwd=root, env=env, timeout=10).returncode, 0)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Run the repository's CI checks without GitLab services or credentials.

The catalog mirrors .gitlab-ci.d/{buildtest,crossbuilds,static_checks}.yml.
Containers are built from the checked-in Dockerfiles, never pulled from QEMU's
CI registry. This module uses only Python's standard library.
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


@dataclasses.dataclass
class Profile:
    name: str
    image: str = 'debian'
    targets: str = ''
    configure: tuple = ()
    checks: tuple = ('check',)
    tcg: tuple = ()
    kind: str = 'build'
    runner: str = 'ubuntu-24.04'
    experimental: bool = False
    advisory: bool = False
    functional: bool = False
    toolchain: str = ''
    env: dict = dataclasses.field(default_factory=dict)


def catalog():
    """Explicit mapping, including optional upstream jobs and extra targets."""
    profiles = {}

    def add(name, **kwargs):
        profiles[name] = Profile(name, **kwargs)

    systems = [
        ('alpine', 'avr loongarch64 mips64 mipsel',
         ('--enable-docs', '--enable-trace-backends=log,simple,syslog')),
        ('ubuntu', 'alpha microblaze mips64el',
         ('--enable-docs', '--enable-rust')),
        ('debian', 'arm hexagon i386 riscv64 sh4eb sparc xtensa',
         ('--with-coroutine=sigaltstack', '--enable-rust')),
        ('fedora', 'mips xtensa m68k riscv32 ppc sparc64',
         ('--disable-gcrypt', '--enable-nettle', '--enable-crypto-afalg',
          '--enable-docs', '--enable-rust')),
        ('centos', 'ppc64 or1k s390x x86_64 rx sh4',
         ('--disable-nettle', '--enable-gcrypt', '--enable-vfio-user-server',
          '--enable-modules', '--enable-trace-backends=dtrace', '--enable-docs')),
        ('opensuse', 's390x x86_64 aarch64', ()),
        # Complete the upstream groups to build every system target in this tree.
        ('extra', 'hppa tricore xtensaeb', ()),
    ]
    for name, arches, flags in systems:
        checks = ['check']
        if name == 'fedora':
            checks += ['check-doc', 'crash-ppc', 'crash-riscv32']
        if name == 'debian':
            checks += ['crash-i386']
        if name == 'centos':
            checks += ['check-block-' + fmt for fmt in (
                'luks', 'nbd', 'parallels', 'qcow2', 'qed', 'raw',
                'vdi', 'vhdx', 'vmdk', 'vpc')]
        add('system-' + name,
            image={'ubuntu': 'ubuntu2404', 'centos': 'centos9',
                   'opensuse': 'opensuse-leap', 'extra': 'debian'}.get(name, name),
            targets=','.join(arch + '-softmmu' for arch in arches.split()),
            configure=flags, checks=tuple(checks), functional=True)

    user_tcg = ('aarch64', 'aarch64_be', 'arm', 'hppa', 'i386', 'm68k',
                'mips', 'mips64', 'mips64el', 'mipsel', 'mips32r5el',
                'mips32r6el', 'mips64r6', 'mips64r6el', 'or1k',
                'ppc64', 'ppc64le', 'riscv64', 's390x', 'x86_64')
    add('user', image='debian-all-test-cross',
        configure=('--disable-tools', '--disable-system',
                   '--target-list-exclude=alpha-linux-user,sh4-linux-user'),
        checks=('check-tcg',),
        tcg=tuple(arch + '-linux-user' for arch in user_tcg))
    add('user-static', image='debian-all-test-cross',
        configure=('--disable-tools', '--disable-system', '--static'),
        checks=('check-tcg',),
        tcg=tuple(arch + '-linux-user' for arch in user_tcg + ('alpha', 'sh4')))
    softmmu = ('alpha', 'arm', 'aarch64', 'i386', 'mips64el',
               'riscv32', 'riscv64', 's390x', 'x86_64')
    add('tcg-system', image='debian-all-test-cross',
        targets=','.join(arch + '-softmmu' for arch in softmmu),
        configure=('--disable-tools', '--enable-debug'),
        checks=('check-tcg',), tcg=tuple(arch + '-softmmu' for arch in softmmu))
    for name, image, targets, flags in [
        ('hexagon', 'debian-hexagon-cross', 'hexagon-linux-user',
         ('--enable-debug-tcg',)),
        ('loongarch64', 'debian-loongarch-cross',
         'loongarch64-linux-user,loongarch64-softmmu', ('--enable-debug',)),
        ('tricore', 'debian-tricore-cross', 'tricore-softmmu',
         ('--disable-fdt', '--enable-debug')),
    ]:
        add('tcg-' + name, image=image, targets=targets,
            configure=('--disable-tools',) + flags, checks=('check-tcg',),
            tcg=tuple(targets.split(',')))
    # The Xtensa Dockerfile only supplies guest compilers. Overlay its /opt on
    # the normal Debian build image rather than pretending it can build QEMU.
    add('tcg-xtensa', targets='xtensa-softmmu,xtensaeb-softmmu',
        toolchain='debian-xtensa-cross', checks=('check-tcg',),
        tcg=('xtensa-softmmu', 'xtensaeb-softmmu'))

    add('clang-system', image='fedora',
        targets='alpha-softmmu,arm-softmmu,m68k-softmmu,mips64-softmmu,s390x-softmmu',
        configure=('--cc=clang', '--cxx=clang++', '--enable-ubsan',
                   '--extra-cflags=-fno-sanitize-recover=undefined'),
        checks=('check-qtest',))
    add('clang-user', image='debian-all-test-cross',
        configure=('--cc=clang', '--cxx=clang++', '--disable-system',
                   '--enable-ubsan', '--extra-cflags=-fno-sanitize-recover=undefined',
                   '--target-list-exclude=alpha-linux-user,microblazeel-linux-user,'
                   'aarch64_be-linux-user,i386-linux-user,m68k-linux-user,'
                   'mipsn32el-linux-user,xtensaeb-linux-user'),
        checks=('check-unit', 'check-tcg'), tcg=('riscv64-linux-user',))
    for name, targets, experimental in [
        ('x86_64', 'x86_64-softmmu', False),
        ('aarch64', 'aarch64-softmmu', True),
        ('ppc64-s390x', 'ppc64-softmmu,s390x-softmmu', True),
    ]:
        add('cfi-' + name, image='fedora', targets=targets,
            configure=('--cc=clang', '--cxx=clang++', '--enable-cfi',
                       '--enable-safe-stack', '--disable-slirp'),
            env={'AR': 'llvm-ar'}, experimental=experimental, functional=True)
    add('tsan-build', image='ubuntu2404',
        targets='x86_64-softmmu,ppc64-softmmu,riscv64-softmmu,x86_64-linux-user',
        configure=('--enable-tsan', '--cc=clang', '--cxx=clang++',
                   '--enable-trace-backends=ust', '--disable-slirp'), checks=())
    add('gcov', image='ubuntu2404',
        targets='aarch64-softmmu,ppc64-softmmu,s390x-softmmu,x86_64-softmmu',
        configure=('--enable-gcov',),
        checks=('check-unit', 'check-softfloat', 'coverage'))
    add('oss-fuzz', image='fedora', kind='fuzz', checks=())
    add('tcg-disabled', image='centos9',
        configure=('--disable-tcg', '--audio-drv-list=', '--with-coroutine=ucontext',
                   '--disable-sdl', '--disable-gtk', '--disable-vnc'),
        checks=('check-unit', 'check-qapi-schema', 'distclean'))
    add('tci', image='debian-all-test-cross',
        targets=','.join(arch + '-softmmu' for arch in (
            'aarch64', 'arm', 'hppa', 'm68k', 'microblaze', 'ppc64', 's390x', 'x86_64')),
        configure=('--enable-tcg-interpreter', '--disable-kvm',
                   '--disable-gtk', '--disable-vnc'),
        checks=('tci-boot', 'check-tcg'),
        tcg=('aarch64-softmmu', 'arm-softmmu', 's390x-softmmu', 'x86_64-softmmu'))
    add('without-defaults', image='centos9',
        configure=('--without-default-devices', '--without-default-features',
                   '--disable-fdt', '--disable-pie', '--disable-qom-cast-debug',
                   '--disable-strip', '--disable-hmp',
                   '--target-list-exclude=aarch64-softmmu,mips64-softmmu,'
                   'mipsel-softmmu,ppc64-softmmu,sh4el-softmmu,xtensa-softmmu,'
                   'x86_64-softmmu'))
    add('multi-target', checks=('multi-target',))
    add('build-units', kind='configure-only', checks=('build-units',))
    add('libvhost-user', image='fedora', kind='libvhost-user', checks=())
    add('tools-docs', configure=('--disable-system', '--disable-user',
                                '--enable-docs', '--enable-tools'),
        checks=('check-unit', 'ctags', 'TAGS', 'cscope', 'source-browser', 'docs-artifact'))
    add('rust-nightly', image='fedora-rust-nightly', targets='aarch64-softmmu',
        configure=('--enable-rust', '--enable-strict-rust-lints'),
        checks=('check', 'check-doc', 'rust-fmt', 'clippy', 'rustdoc'), advisory=True)
    add('migration-compat-x86_64', image='opensuse-leap',
        targets='x86_64-softmmu,aarch64-softmmu',
        checks=('migration',), advisory=True)
    add('functional-flaky', targets=','.join(arch + '-softmmu' for arch in (
        'aarch64', 'arm', 'mips64el', 'ppc64', 'rx', 's390x', 'sh4', 'x86_64')),
        checks=(), functional=True, experimental=True, advisory=True,
        env={'QEMU_TEST_FLAKY_TESTS': '1'})

    for arch in ('arm64', 'ppc64el', 'riscv64', 's390x'):
        for mode in ('system', 'user', 'kvm-only'):
            flags = []
            if mode == 'kvm-only':
                flags += ['--disable-tcg', '--enable-kvm', '--disable-tools']
                flags += {
                    'arm64': ['--without-default-features'],
                    'riscv64': ['--without-default-features'],
                    'ppc64el': ['--without-default-devices'],
                    's390x': ['--enable-trace-backends=ftrace'],
                }[arch]
            add('cross-' + arch + '-' + mode, image='debian-' + arch + '-cross',
                configure=tuple(flags), checks=(), kind='cross-' + mode)
    for arch in ('amd64', 'arm64'):
        add('cross-' + arch + '-xen-only', image='debian-' + arch + '-cross',
            configure=('--disable-tcg', '--disable-kvm', '--enable-xen',
                       '--disable-tools'),
            checks=(), kind='cross-accel')
    add('cross-win64-system', image='fedora-win64-cross', kind='cross-system',
        configure=('--enable-fdt=internal', '--disable-plugins', '--enable-slirp',
                   '--enable-tools'), checks=('installer',))
    for bits in ('64', '32'):
        flags = ('--static', '--cpu=wasm64', '--disable-tools',
                 '--enable-debug', '--enable-tcg-interpreter')
        if bits == '32':
            flags += ('--wasm64-32bit-address-limit',)
        add('wasm64-' + bits + 'bit', image='emsdk-wasm64-cross', kind='wasm',
            configure=flags, checks=())

    # Public GitHub arm64 runners replace the upstream dedicated AArch64 host.
    for variant, flags, checks in [
        ('default', (), ('check',)),
        ('debug', ('--enable-debug',), ('check', 'check-tcg')),
        ('static-user', ('--static', '--disable-system'), ('check-tcg',)),
        ('minimal', ('--disable-user', '--without-default-devices',
                     '--without-default-features'), ('check',)),
        ('clang', ('--cc=clang', '--cxx=clang++', '--enable-ubsan'), ('check',)),
        ('tci', ('--enable-tcg-interpreter',), ('check',)),
        ('notcg', ('--disable-tcg', '--with-devices-aarch64=minimal'), ('check',)),
    ]:
        add('arm64-' + variant, image='ubuntu2404', runner='ubuntu-24.04-arm',
            configure=flags, checks=checks,
            tcg=('aarch64-linux-user',) if variant in ('debug', 'static-user') else ())

    for release in ('15', '26'):
        targets = 'aarch64-softmmu,i386-softmmu,x86_64-softmmu'
        if release == '15':
            targets += ',arm-softmmu'
        add('macos-' + release, image='', runner='macos-' + release, targets=targets,
            configure=('--disable-plugins',), checks=('check-unit', 'check-tcg'),
            tcg=tuple(targets.split(',')), kind='native')
    add('windows-mingw64', image='', runner='windows-2022', kind='native',
        configure=('--disable-system', '--enable-tools', '--enable-rust',
                   '-Ddebug=false', '-Doptimization=0'))
    return profiles


def selected_profiles(selection='all', experimental=False):
    profiles = catalog()
    if selection in ('riscv64-checks', 'python', 'commits'):
        return []  # Existing dedicated workflow job retains its check name.
    if selection != 'all':
        if selection not in profiles:
            raise ValueError(f'Unknown profile: {selection}')
        return [profiles[selection]]
    return [p for p in profiles.values() if experimental or not p.experimental]


def emit_matrix(selection, experimental):
    selected = selected_profiles(selection, experimental)
    outputs = {}
    for platform in ('linux', 'native'):
        rows = []
        for p in selected:
            if (p.kind == 'native') != (platform == 'native'):
                continue
            row = {'profile': p.name, 'runner': p.runner, 'advisory': p.advisory}
            if platform == 'native':
                row['shell'] = (r'C:\msys64\usr\bin\bash.exe --login -eo pipefail {0}'
                                if p.runner.startswith('windows') else 'bash')
            rows.append(row)
        outputs[platform] = json.dumps({'include': rows}, separators=(',', ':'))
        outputs['has_' + platform] = str(bool(rows)).lower()
    for key, value in outputs.items():
        print(f'{key}={value}')
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as stream:
            for key, value in outputs.items():
                stream.write(f'{key}={value}\n')


class Runner:
    """Run independent phases, preserving failure status and separate logs."""

    def __init__(self, profile, root=ROOT):
        self.profile = profile
        self.root = Path(root).resolve()
        self.build = self.root / 'build-ci'
        self.logs = self.build / 'ci-logs'
        self.logs.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.jobs = os.environ.get('CI_JOBS', '2')
        self.make = os.environ.get('MAKE', 'make')
        self.env = dict(os.environ, **profile.env)
        self.env.update({
            'QEMU_TEST_ALLOW_UNTRUSTED_CODE': '1',
            'QEMU_TEST_CACHE_DIR': str(self.build / 'functional-cache'),
        })
        # Never set GITLAB_CI: individual tests retain their real platform context.

    def run(self, name, command, cwd=None, env=None, record=True):
        print(f'::group::{name}', flush=True)
        print(shlex.join(str(part) for part in command), flush=True)
        status = 1
        with open(self.logs / (name + '.log'), 'w', encoding='utf-8') as log:
            try:
                with subprocess.Popen(
                    [str(part) for part in command], cwd=cwd or self.build,
                    env=env or self.env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, errors='replace',
                ) as process:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                    status = process.wait()
            except OSError as error:
                print(error, file=log)
                print(error, flush=True)
        if record:
            self.results.append({'phase': name, 'exit_code': status})
            self.logs.mkdir(parents=True, exist_ok=True)
            (self.logs / 'results.json').write_text(
                json.dumps(self.results, indent=2) + '\n', encoding='utf-8')
        print('::endgroup::', flush=True)
        return status == 0

    def make_check(self, name):
        return self.run(name, [self.make, '-j' + self.jobs, name,
                              'MTESTARGS=--logbase ' + name + ' --print-errorlogs'])

    def configure_command(self):
        p = self.profile
        args = ['bash', (self.root / 'configure').as_posix(), '--enable-werror',
                '--disable-docs', '--disable-debug-info', '--disable-containers']
        if p.kind != 'wasm' and not p.runner.startswith('windows'):
            args += ['--enable-fdt=system']
        if p.targets:
            args += ['--target-list=' + p.targets]
        if p.kind.startswith('cross-'):
            args += shlex.split(os.environ.get('QEMU_CONFIGURE_OPTS', ''))
            if p.kind == 'cross-system':
                exclude = ('arm i386 microblaze mips mipsel mips64 ppc '
                           'riscv32 sh4 sparc xtensa').split()
                if p.name == 'cross-win64-system':
                    exclude += ('alpha avr hppa m68k or1k rx sh4eb sparc64 '
                                'tricore xtensaeb').split()
                args += ['--disable-user', '--target-list-exclude=' + ','.join(
                    arch + '-softmmu' for arch in sorted(set(exclude)))]
            elif p.kind == 'cross-user':
                exclude = ('aarch64_be alpha m68k microblazeel or1k ppc '
                           'sparc xtensa').split()
                args += ['--disable-system', '--target-list-exclude=' + ','.join(
                    arch + '-linux-user' for arch in exclude)]
        args += p.configure
        if p.kind == 'wasm':
            args.insert(0, 'emconfigure')
        return args

    def verify_tcg(self):
        if not self.profile.tcg:
            return
        info = self.build / 'meson-info'
        tests = json.loads((info / 'intro-tests.json').read_text(encoding='utf-8'))
        targets = json.loads((info / 'intro-targets.json').read_text(encoding='utf-8'))
        counts = {}
        for target in self.profile.tcg:
            registered = [test for test in tests if 'qemu:tcg-' + target in test['suite']]
            guests = [item for item in targets if item['name'].startswith(target + '-')
                      and item['name'].endswith('.test')]
            if not registered or not guests:
                raise ValueError(f'{target}: missing guest compiler or TCG registration')
            counts[target] = {'tests': len(registered), 'guests': len(guests)}
        (self.logs / 'tcg-registration.json').write_text(
            json.dumps(counts, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(counts, indent=2), flush=True)

    def phase(self, name):
        if name.startswith('crash-'):
            return self.run(name, [self.build / 'run', self.root / 'scripts/device-crash-test',
                                   '-q', '--tcg-only', './qemu-system-' + name[6:]])
        if name == 'coverage':
            return self.run(name, ['gcovr', '--xml-pretty', '--exclude-unreachable-branches',
                                   '--print-summary', '-o', 'coverage.xml',
                                   '--root', self.root, '.'])
        if name == 'build-units':
            return self.run(name, [sys.executable, self.root / '.gitlab-ci.d/check-units.py',
                                   self.build / 'compile_commands.json'])
        if name == 'multi-target':
            return (self.run('build-multi-target', [self.make, '-j' + self.jobs,
                                                   'tests/unit/test-multi-target']) and
                    self.run(name, [self.build / 'tests/unit/test-multi-target']))
        if name == 'rust-fmt':
            return self.run(name, [self.build / 'pyvenv/bin/meson', 'devenv', '-w', self.root,
                                   os.environ.get('CARGO', 'cargo'), 'fmt', '--check'])
        if name == 'tci-boot':
            if not self.run('build-tci-boot', [self.make, '-j' + self.jobs,
                                              'tests/qtest/boot-serial-test',
                                              'tests/qtest/cdrom-test']):
                return False
            passed = True
            for target in self.profile.targets.split(','):
                arch = target.removesuffix('-softmmu')
                env = dict(self.env, QTEST_QEMU_BINARY='./qemu-system-' + arch)
                for test in ('boot-serial', 'cdrom'):
                    passed = self.run(arch + '-' + test,
                                      [self.build / ('tests/qtest/' + test + '-test')],
                                      env=env) and passed
            return passed
        if name == 'docs-artifact':
            return self.run(name, [self.make, 'install',
                                   'DESTDIR=' + str(self.build / 'docs-install')])
        if name == 'source-browser':
            if not self.make_check('gtags'):
                return False
            return self.run(name, ['htags', '-anT', '--tree-view=filetree', '-m',
                                   'qemu_init', '-t', 'QEMU source'], cwd=self.root)
        if name == 'migration':
            return self.migration()
        # distclean would delete the logs and JUnit files before artifact upload.
        if name == 'distclean':
            archive = self.root / 'ci-results'
            shutil.copytree(self.logs, archive / 'ci-logs', dirs_exist_ok=True)
            shutil.copytree(self.build / 'meson-logs', archive / 'meson-logs',
                            dirs_exist_ok=True)
            result = self.run(name, [self.make, 'distclean'])
            return result
        return self.make_check(name)

    def migration(self):
        version = (self.root / 'VERSION').read_text(encoding='utf-8').strip()
        if re.search(r'\.0$', version):
            print('SKIP migration compatibility on a round release, as in GitLab CI.')
            return True
        match = re.match(r'(\d+)\.(\d+)\.', version)
        if not match:
            raise ValueError('Unrecognized VERSION: ' + version)
        tag = f'v{match[1]}.{match[2]}.0'
        previous = self.build / 'previous-source'
        if not self.run('fetch-previous', ['git', 'fetch', '--depth=1', '--no-tags',
                                         'https://gitlab.com/qemu-project/qemu.git',
                                         'refs/tags/' + tag], cwd=self.root):
            return False
        if not self.run('checkout-previous', ['git', 'worktree', 'add', '--detach',
                                             previous, 'FETCH_HEAD'], cwd=self.root):
            return False
        oldbuild = previous / 'build'
        oldbuild.mkdir()
        if not self.run('configure-previous', ['bash', previous / 'configure',
                         '--target-list=x86_64-softmmu,aarch64-softmmu',
                         '--disable-docs', '--disable-werror', '--disable-containers'],
                        cwd=oldbuild):
            return False
        if not self.run('build-previous', [self.make, '-j' + self.jobs, 'all', 'check-build'],
                        cwd=oldbuild):
            return False
        passed = True
        for direction, variable in [('old-to-new', 'QTEST_QEMU_BINARY_SRC'),
                                    ('new-to-old', 'QTEST_QEMU_BINARY_DST')]:
            env = dict(self.env, QTEST_QEMU_BINARY=str(self.build / 'qemu-system-x86_64'))
            env[variable] = str(oldbuild / 'qemu-system-x86_64')
            passed = self.run(direction, [oldbuild / 'tests/qtest/migration-test'],
                              cwd=oldbuild, env=env) and passed
        return passed

    def fuzz(self):
        env = dict(self.env, CC='clang', CXX='clang++', CFLAGS='-fsanitize=address',
                   LSAN_OPTIONS='suppressions=' + str(self.root / 'scripts/lsan_suppressions.txt'))
        if not self.run('build-fuzz', ['bash', self.root / 'scripts/oss-fuzz/build.sh'],
                        cwd=self.root, env=env):
            return False
        env['ASAN_OPTIONS'] = 'fast_unwind_on_malloc=0'
        fuzzers = list((self.root / 'build-oss-fuzz/DEST_DIR').glob('qemu-fuzz-*-target-*'))
        if not fuzzers:
            raise ValueError('No fuzz targets were built')
        passed = True
        for fuzzer in fuzzers:
            if 'slirp' not in fuzzer.name:
                passed = self.run(fuzzer.name, [fuzzer, '-runs=1', '-seed=1'],
                                  cwd=self.root, env=env) and passed
        return passed

    def execute(self, functional=True):
        p = self.profile
        if p.kind == 'fuzz':
            return self.fuzz()
        if p.kind == 'libvhost-user':
            return (self.run('configure', ['meson', 'setup', str(self.build / 'libvhost-user'),
                                          str(self.root / 'subprojects/libvhost-user')]) and
                    self.run('build', ['ninja', '-C', str(self.build / 'libvhost-user')]))
        packages = Path('/packages.txt')
        if packages.is_file():
            shutil.copy2(packages, self.logs / 'packages.txt')
        if not self.run('configure', self.configure_command()):
            return False
        try:
            self.verify_tcg()
        except (ValueError, OSError) as error:
            (self.logs / 'tcg-error.log').write_text(str(error) + '\n', encoding='utf-8')
            print(f'::error::{error}', flush=True)
            return False
        if p.name.startswith('cfi-'):
            if not self.run('limit-link-memory', [self.build / 'pyvenv/bin/meson',
                                                  'configure', '.',
                                                  '-Dbackend_max_links=1']):
                return False
        if p.kind != 'configure-only' and p.name != 'multi-target':
            # Cross builds only compile the tests; never claim to run host binaries.
            command = [self.make, '-j' + self.jobs, 'all', 'check-build']
            if p.kind == 'wasm':
                command = ['emmake', self.make, '-j' + self.jobs, 'all']
            if not self.run('build', command):
                return False
        checks = list(p.checks)
        if p.functional and functional:
            checks += ['check-functional']
        passed = True
        for name in checks:
            passed = self.phase(name) and passed
        return passed


def docker_run(profile, functional=True):
    """No registry login, Docker socket mount, privileged mode, or secrets."""
    logs = ROOT / 'build-ci/ci-logs'
    logs.mkdir(parents=True, exist_ok=True)
    runner = Runner(profile)
    tag = 'qemu-ci-' + profile.name
    dockerfile = ROOT / 'tests/docker/dockerfiles' / (profile.image + '.docker')

    def build_image(image_tag, content, phase):
        path = logs / (phase + '.docker')
        path.write_text(content, encoding='utf-8')
        # Empty build context avoids sending build output or untracked files.
        with open(path, 'rb') as stream:
            command = ['docker', 'build', '--progress=plain', '-t', image_tag]
            if profile.image != 'alpine':
                command += ['--build-arg', 'USER=ci-runner',
                            '--build-arg', 'UID=' + str(os.getuid())]
            command += ['-']
            with open(logs / (phase + '.log'), 'w', encoding='utf-8') as log:
                process = subprocess.Popen(command, stdin=stream, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                return process.wait() == 0

    content = dockerfile.read_text(encoding='utf-8')
    if profile.image == 'alpine':
        content += (f'\nRUN addgroup -g {os.getgid()} ci-runner && '
                    f'adduser -D -u {os.getuid()} -G ci-runner ci-runner\n')
    if profile.image == 'debian-all-test-cross':
        # These targets now use i686-linux-gnu-gcc, whereas the upstream
        # omnibus image does not install it. Keep i386 coverage explicit.
        content += ('\nRUN apt-get update && apt-get install -y --no-install-recommends '
                    'g++ gcc-i686-linux-gnu libc6-dev-i386-cross\n')
    if not build_image(tag, content, 'image'):
        return False
    if profile.toolchain:
        tools = tag + '-tools'
        toolsfile = ROOT / 'tests/docker/dockerfiles' / (profile.toolchain + '.docker')
        if not build_image(tools, toolsfile.read_text(encoding='utf-8'), 'toolchain'):
            return False
        release = '/opt/2020.07'
        overlay = (f'FROM {tools} AS tools\nFROM {tag}\n'
                   f'COPY --from=tools {release} {release}\n'
                   'ENV PATH="' + ':'.join(
                       release + '/xtensa-' + cpu + '-elf/bin' for cpu in
                       ('dc232b', 'dc233c', 'de233_fpu', 'dsp3400', 'test_kc705_be'))
                   + ':$PATH"\n')
        if not build_image(tag, overlay, 'overlay'):
            return False
    # Run unprivileged. Qtest's 9p tests expect a non-root user. HOME must be
    # writable even for images that do not supply a runner account.
    container_home = ROOT / 'build-ci/container-home'
    container_home.mkdir(exist_ok=True)
    command = ['docker', 'run', '--rm', '--init', '--cpus=2',
               '--user', f'{os.getuid()}:{os.getgid()}',
               '-v', f'{ROOT}:{ROOT}', '-w', str(ROOT),
               '-e', 'HOME=' + str(container_home),
               '-e', 'CI_JOBS=2', '-e', 'GITHUB_ACTIONS=true',
               '-e', 'CI=true', tag, 'python3', 'scripts/ci/github_ci.py',
               'run', profile.name, '--inside']
    if not functional:
        command += ['--no-functional']
    return runner.run('container', command, cwd=ROOT, record=False)


def commit_range(event, root=ROOT):
    """Check the contribution, not every commit since upstream QEMU master."""
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()

    if 'pull_request' in event:
        pr = event['pull_request']
        head = pr['head']['sha']
        base = git('merge-base', pr['base']['sha'], head)
    else:
        head = event.get('after') or git('rev-parse', 'HEAD')
        base = event.get('before', '')
        if not base or set(base) == {'0'}:
            branch = event.get('repository', {}).get('default_branch', 'master')
            try:
                base = git('merge-base', 'refs/remotes/origin/' + branch, head)
            except subprocess.CalledProcessError:
                base = ''
            if base == head or not base:
                # Empty tree handles a repository's first commit without
                # inventing a dependency on an external upstream remote.
                parents = git('rev-list', '--parents', '-n', '1', head).split()
                base = parents[1] if len(parents) > 1 else ''
    return base, head


def check_commits(mode):
    with open(os.environ['GITHUB_EVENT_PATH'], encoding='utf-8') as stream:
        base, head = commit_range(json.load(stream))
    commits = subprocess.check_output(
        ['git', 'rev-list', '--reverse', f'{base}..{head}' if base else head],
        cwd=ROOT, text=True).splitlines()
    passed = True
    for commit in commits:
        print('Checking commit ' + commit, flush=True)
        if mode == 'patch':
            patch = subprocess.check_output(['git', 'show', '--format=email', commit], cwd=ROOT)
            result = subprocess.run(['perl', 'scripts/checkpatch.pl', '--terse', '-'],
                                    cwd=ROOT, input=patch)
            passed = result.returncode == 0 and passed
        else:
            message = subprocess.check_output(['git', 'show', '-s', '--format=%B', commit],
                                              cwd=ROOT, text=True)
            signoffs = [line for line in message.splitlines() if 'Signed-off-by:' in line]
            if not signoffs or any('localhost' in line for line in signoffs):
                print(f'::error::{commit}: missing or invalid Signed-off-by')
                passed = False
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    matrix = subparsers.add_parser('matrix')
    matrix.add_argument('--profile', default='all')
    matrix.add_argument('--experimental', action='store_true')
    run = subparsers.add_parser('run')
    run.add_argument('profile', choices=sorted(catalog()))
    run.add_argument('--inside', action='store_true')
    run.add_argument('--no-functional', action='store_true')
    commits = subparsers.add_parser('commits')
    commits.add_argument('mode', choices=('dco', 'patch'))
    subparsers.add_parser('list')
    args = parser.parse_args()
    if args.command == 'matrix':
        emit_matrix(args.profile, args.experimental)
        return 0
    if args.command == 'list':
        for profile in catalog().values():
            print(json.dumps(dataclasses.asdict(profile)))
        return 0
    if args.command == 'commits':
        return 0 if check_commits(args.mode) else 1
    profile = catalog()[args.profile]
    if args.inside or profile.kind == 'native':
        passed = Runner(profile).execute(functional=not args.no_functional)
    else:
        passed = docker_run(profile, functional=not args.no_functional)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())

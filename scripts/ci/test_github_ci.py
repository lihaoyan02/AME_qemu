#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Fast, offline tests for the GitHub CI adapter; never build or run QEMU."""

import contextlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import github_ci as ci


class CatalogTests(unittest.TestCase):
    def test_all_system_targets_are_built(self):
        expected = {path.stem for path in (ci.ROOT / 'configs/targets').glob('*-softmmu.mak')}
        actual = {target for p in ci.catalog().values() if p.name.startswith('system-')
                  for target in p.targets.split(',')}
        self.assertEqual(expected, actual)

    def test_all_registered_tcg_targets_are_required(self):
        expected = set()
        for path in (ci.ROOT / 'tests/tcg').rglob('meson.build'):
            expected.update(re.findall(r"'([\w-]+-(?:linux-user|softmmu))':\s*\{",
                                       path.read_text(encoding='utf-8')))
        expected -= {'multiarch-linux-user', 'multiarch-softmmu'}
        actual = {target for profile in ci.catalog().values() for target in profile.tcg}
        self.assertEqual(expected, actual)

    def test_only_existing_images_and_targets(self):
        for profile in ci.catalog().values():
            if profile.image:
                path = ci.ROOT / 'tests/docker/dockerfiles' / (profile.image + '.docker')
                self.assertTrue(path.is_file(), profile.name)
            for target in profile.targets.split(',') if profile.targets else []:
                path = ci.ROOT / 'configs/targets' / (target + '.mak')
                self.assertTrue(path.is_file(), (profile.name, target))

    def test_experimental_is_opt_in(self):
        normal = {p.name for p in ci.selected_profiles()}
        full = {p.name for p in ci.selected_profiles(experimental=True)}
        self.assertEqual(full - normal,
                         {'functional-flaky', 'cfi-aarch64', 'cfi-ppc64-s390x'})
        self.assertEqual(ci.selected_profiles('cfi-aarch64')[0].name, 'cfi-aarch64')

    def test_selection_and_empty_platforms(self):
        with self.assertRaises(ValueError):
            ci.selected_profiles('unknown-profile')
        for name in ('python', 'commits', 'riscv64-checks'):
            self.assertEqual(ci.selected_profiles(name), [])
        with contextlib.redirect_stdout(io.StringIO()), patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'output'
                with patch.dict(os.environ, {'GITHUB_OUTPUT': str(output)}):
                    ci.emit_matrix('windows-mingw64', False)
                values = dict(line.split('=', 1) for line in output.read_text().splitlines())
                self.assertEqual(values['has_linux'], 'false')
                self.assertEqual(values['has_native'], 'true')
                rows = json.loads(values['native'])['include']
                self.assertEqual(rows[0]['runner'], 'windows-2022')


class RunnerTests(unittest.TestCase):
    def test_independent_tests_continue_and_failure_is_preserved(self):
        profile = ci.Profile('test', kind='configure-only', checks=('check', 'check-tcg'))
        with tempfile.TemporaryDirectory() as directory:
            runner = ci.Runner(profile, root=directory)
            with patch.object(runner, 'run', return_value=True), \
                    patch.object(runner, 'phase', side_effect=[False, True]) as phase:
                self.assertFalse(runner.execute())
                self.assertEqual([call.args[0] for call in phase.call_args_list],
                                 ['check', 'check-tcg'])

    def test_functional_can_be_disabled_without_disabling_other_tests(self):
        profile = ci.Profile('test', kind='configure-only', functional=True)
        with tempfile.TemporaryDirectory() as directory:
            runner = ci.Runner(profile, root=directory)
            with patch.object(runner, 'run', return_value=True), \
                    patch.object(runner, 'phase', return_value=True) as phase:
                self.assertTrue(runner.execute(functional=False))
                self.assertEqual([call.args[0] for call in phase.call_args_list], ['check'])

    def test_subprocess_failure_is_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = ci.Runner(ci.Profile('test'), root=directory)
            with contextlib.redirect_stdout(io.StringIO()):
                passed = runner.run('failing-test', [sys.executable, '-c',
                                                     'print("failure evidence"); exit(3)'])
            self.assertFalse(passed)
            self.assertIn('failure evidence', (runner.logs / 'failing-test.log').read_text())
            results = json.loads((runner.logs / 'results.json').read_text())
            self.assertEqual(results, [{'phase': 'failing-test', 'exit_code': 3}])

    def test_container_wrapper_preserves_inner_phase_results(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = ci.Runner(ci.Profile('test'), root=directory)
            result_file = runner.logs / 'results.json'
            evidence = '[{"phase": "check", "exit_code": 2}]\n'
            result_file.write_text(evidence)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(runner.run('container', [sys.executable, '-c', 'exit(1)'],
                                            record=False))
            self.assertEqual(result_file.read_text(), evidence)

    def test_missing_tcg_registration_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = ci.Runner(ci.Profile('test', tcg=('riscv64-softmmu',)), root=directory)
            info = runner.build / 'meson-info'
            info.mkdir()
            for name in ('tests', 'targets'):
                (info / ('intro-' + name + '.json')).write_text('[]')
            with self.assertRaisesRegex(ValueError, 'missing guest compiler'):
                runner.verify_tcg()

    def test_cross_build_uses_image_compiler_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'QEMU_CONFIGURE_OPTS': '--cross-prefix=aarch64-linux-gnu-'}):
                command = ci.Runner(ci.catalog()['cross-arm64-system'], root=directory).configure_command()
            self.assertIn('--cross-prefix=aarch64-linux-gnu-', command)
            self.assertIn('--disable-user', command)
            self.assertIn('--disable-containers', command)

    def test_windows_does_not_require_uninstalled_fdt(self):
        with tempfile.TemporaryDirectory() as directory:
            command = ci.Runner(ci.catalog()['windows-mingw64'], root=directory).configure_command()
            self.assertNotIn('--enable-fdt=system', command)


class CommitRangeTests(unittest.TestCase):
    def test_pr_uses_merge_base_and_head_instead_of_synthetic_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                return subprocess.check_output(['git', '-c', 'user.name=Test',
                                                '-c', 'user.email=test@example.com',
                                                *args], cwd=root, text=True,
                                               stderr=subprocess.DEVNULL).strip()

            git('init', '-b', 'master')
            git('commit', '--allow-empty', '-m', 'base')
            base = git('rev-parse', 'HEAD')
            git('checkout', '-b', 'feature')
            git('commit', '--allow-empty', '-m', 'contribution')
            head = git('rev-parse', 'HEAD')
            git('checkout', 'master')
            git('commit', '--allow-empty', '-m', 'new base commit')
            target = git('rev-parse', 'HEAD')
            self.assertEqual(ci.commit_range({'pull_request': {
                'base': {'sha': target}, 'head': {'sha': head}}}, root=root), (base, head))
            git('remote', 'add', 'origin', str(root))
            git('update-ref', 'refs/remotes/origin/master', target)
            self.assertEqual(ci.commit_range({'before': '0' * 40, 'after': head,
                                              'repository': {'default_branch': 'master'}},
                                             root=root), (base, head))


if __name__ == '__main__':
    unittest.main()

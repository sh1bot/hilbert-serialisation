import os
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from pixel_bus import coordinates, count_transitions, load_image, pack_chunks, analyze, transfer_layout, extend_lanes, sign_magnitude_delta, decode_sign_magnitude, format_summary, fold_signed_bytes, fold_biased_chroma


class MeasurementTests(unittest.TestCase):
    def test_orders(self):
        for side in (1, 2, 4, 8, 16):
            for order in ('raster', 'z', 'hilbert'):
                c = coordinates(side, side, order)
                self.assertEqual(len(set(map(tuple, c))), side*side)
                if order == 'hilbert':
                    self.assertTrue(np.all(np.abs(np.diff(c, axis=0)).sum(axis=1) == 1))
        self.assertEqual(coordinates(2, 2, 'hilbert').tolist(), [[0,0],[0,1],[1,1],[1,0]])
        self.assertEqual(coordinates(2, 2, 'z').tolist(), [[0,0],[1,0],[0,1],[1,1]])
        for order in ('raster', 'z', 'hilbert'):
            self.assertEqual(set(map(tuple, coordinates(3, 7, order))), {(x,y) for x in range(3) for y in range(7)})

    def test_count_against_scalar_reference(self):
        rng = np.random.default_rng(7)
        words = rng.integers(0, 256, (23, 6), dtype=np.uint8)
        indices = rng.permutation(23)
        for lag in (1, 2, 7, 23, 30):
            expected = [[sum(((int(words[indices[t], lane]) ^ int(words[indices[t-lag], lane])) >> bit) & 1
                             for t in range(lag, 23)) for bit in range(8)] for lane in range(6)]
            self.assertEqual(count_transitions(words, indices, lag, batch_size=3)['by_lane_lsb_first'], expected)

    def test_chunk_lane_mapping_and_edges(self):
        p = np.arange(15, dtype=np.uint8).reshape(3,5)
        words, shapes, gx, gy = pack_chunks({'R': p}, {'R': (1,1)}, 5,3,(2,2))
        self.assertEqual((gx,gy), (3,2))
        self.assertEqual(words['R'][0].tolist(), [0,1,5,6])
        self.assertEqual(words['R'][-1].tolist(), [14,14,14,14])

    def test_macroblock_raster_and_cross_boundary_lag(self):
        coords, _ = transfer_layout(4, 4, 'hilbert', (1,1), (2,2))
        self.assertEqual(coords[:8].tolist(),
                         [[0,0],[0,1],[1,1],[1,0],[2,0],[2,1],[3,1],[3,0]])
        plane = np.arange(15, dtype=np.uint8).reshape(3,5)
        report = analyze({'Y': plane}, {'Y': (1,1)}, {'width':5,'height':3},
                         [(1,1)], [1,3], (2,2))
        self.assertEqual(report['results'][0]['cycles'], 24)
        self.assertEqual(report['results'][0]['comparisons'], 23)
        self.assertEqual(report['results'][6]['comparisons'], 21)
        with self.assertRaises(ValueError):
            analyze({'Y': plane}, {'Y': (1,1)}, {'width':5,'height':3}, [(2,2)], [1], (3,4))

    def test_extended_curve_matches_full_resolution_curve(self):
        plane = np.arange(64, dtype=np.uint8).reshape(8,8)
        words, shapes, gx, gy = pack_chunks({'Y':plane}, {'Y':(1,1)},8,8,(2,2))
        for order in ('raster', 'z', 'hilbert'):
            coords, local = transfer_layout(gx,gy,order,(2,2),(8,8))
            extended = extend_lanes(words['Y'],shapes['Y'],coords,gx,local,(2,2),(1,1),order)
            indices = coords[:,1]*gx+coords[:,0]
            if order != 'raster':
                full = coordinates(8,8,order)
                np.testing.assert_array_equal(extended[indices].ravel(), plane[full[:,1],full[:,0]])
            else:
                np.testing.assert_array_equal(extended, words['Y'])
            if order == 'hilbert':
                permutations = {tuple((row // 8 % 2)*2 + row % 2) for row in extended}
                self.assertEqual(len(permutations), 4)
        # Subsampled channels retain all their own sample lanes.
        chroma = plane[::2,::2]
        words, shapes, gx, gy = pack_chunks({'U':chroma}, {'U':(2,2)},8,8,(4,4))
        coords, local = transfer_layout(gx,gy,'hilbert',(4,4),(8,8))
        ext = extend_lanes(words['U'],shapes['U'],coords,gx,local,(4,4),(2,2),'hilbert')
        np.testing.assert_array_equal(np.sort(ext.ravel()), np.sort(chroma.ravel()))
        with self.assertRaises(ValueError):
            extend_lanes(words['U'],shapes['U'],coords,gx,local,(4,2),(2,2),'hilbert')

    def test_fold_mapping_and_chroma_center(self):
        values = np.arange(256, dtype=np.uint8)
        expected = np.array([d ^ (127 if d & 128 else 0) for d in range(256)], dtype=np.uint8)
        np.testing.assert_array_equal(fold_signed_bytes(values), expected)
        np.testing.assert_array_equal(fold_signed_bytes(fold_signed_bytes(values)), values)
        signed = np.arange(-4,5).astype(np.uint8)
        self.assertEqual(fold_signed_bytes(signed).tolist(), [131,130,129,128,0,1,2,3,4])
        self.assertEqual(fold_biased_chroma([126,127,128,129,130]).tolist(), [129,128,0,1,2])

    def test_delta_all_byte_pairs_and_resets(self):
        a, b = np.indices((256,256), dtype=np.uint16)
        stream = np.stack([a.ravel(), b.ravel()]).astype(np.uint8)
        encoded = sign_magnitude_delta(stream, 2)
        np.testing.assert_array_equal(decode_sign_magnitude(encoded, 2), stream)
        self.assertEqual(len(np.unique(encoded[1, :256])), 256)
        special = np.array([[255], [0], [128], [128], [127]], dtype=np.uint8)
        self.assertEqual(sign_magnitude_delta(special, 5).ravel().tolist(), [128,1,255,0,128])
        for interval in (1,2,7,100):
            data = np.random.default_rng(3).integers(0,256,(53,6),dtype=np.uint8)
            np.testing.assert_array_equal(decode_sign_magnitude(sign_magnitude_delta(data, interval), interval), data)

    def test_delta_report_matches_explicit_boundary_stream(self):
        plane = np.array([[1,2,3,4]], dtype=np.uint8)
        report = analyze({'Y':plane}, {'Y':(1,1)}, {'width':4,'height':1}, [(1,1)], [1,2], (2,1))
        encoded = np.array([[1],[1],[3],[1]], dtype=np.uint8)
        for lag, offset in ((1,0),(2,6)):
            raw, delta = report['results'][offset:offset+2]
            expected = count_transitions(encoded,np.arange(4),lag)
            self.assertEqual(delta['channels']['Y']['by_lane_lsb_first'], expected['by_lane_lsb_first'])
            self.assertEqual(delta['comparisons'], 4-lag)
            self.assertEqual(delta['encoding'], 'delta-sm')
            self.assertEqual(raw['encoding'], 'raw')

    def test_summary_uses_raw_raster_baseline(self):
        report = {"results": [dict(order=order, encoding=encoding, total=total,
                  chunk=[2,2], lag=1, bus_bits=96, fold_only={"total":70})
                  for order, encoding, total in [
                      ('raster','raw',100), ('raster','delta-sm',80),
                      ('z','raw',90), ('z','delta-sm',60),
                      ('hilbert','raw',110), ('hilbert','delta-sm',50)]]}
        summary = format_summary(report)
        self.assertEqual(len(summary.splitlines()), 4)
        self.assertNotIn('delta-sm', summary)
        self.assertNotIn('encoding', summary)
        self.assertIn('transitions', summary.splitlines()[0])
        self.assertIn('vs raster', summary.splitlines()[0])
        self.assertIn('with sm-delta', summary.splitlines()[0])
        self.assertIn('fold only', summary.splitlines()[0])
        self.assertIn('30.00%', summary.splitlines()[2])
        self.assertEqual(summary.splitlines()[2].split()[5], '90')
        self.assertIn('10.00%', summary.splitlines()[2])
        self.assertIn('40.00%', summary.splitlines()[2])
        self.assertIn('-10.00%', summary.splitlines()[3])
        self.assertIn('50.00%', summary.splitlines()[3])
        report['results'][0]['total'] = 0
        self.assertIn('n/a', format_summary(report))

    def test_json_uses_raw_raster_for_every_encoding(self):
        plane = np.random.default_rng(17).integers(0,256,(8,8),dtype=np.uint8)
        report = analyze({'Y':plane}, {'Y':(1,1)}, {'width':8,'height':8},
                         [(1,1),(2,2)], [1,3], (4,4))
        for start in range(0,len(report['results']),6):
            rows = report['results'][start:start+6]
            baseline = rows[0]['total']
            for row in rows:
                expected = 100*(baseline-row['total'])/baseline
                self.assertEqual(row['baseline_raw_raster_total'], baseline)
                self.assertAlmostEqual(row['reduction_vs_raw_percent'], expected)
                self.assertAlmostEqual(row['reduction_vs_raster_percent'], expected)
                self.assertAlmostEqual(row['channels']['Y']['reduction_vs_raw_percent'], expected)

    def test_cli_multiple_macroblocks_and_lags(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, output = Path(tmp)/'image.png', Path(tmp)/'results.json'
            Image.fromarray(np.arange(64,dtype=np.uint8).reshape(8,8)).save(path)
            run = subprocess.run([sys.executable, str(Path(__file__).with_name('pixel_bus.py')),
                str(path), '--chunk','2x2','--macroblock','4x4','--macroblock','8x8',
                '--lag','1','--lag','3','--json',str(output)], check=True, capture_output=True, text=True)
            report = json.loads(output.read_text())
            self.assertEqual(report['macroblocks'], [[4,4],[8,8]])
            self.assertEqual(len(report['results']), 24)
            self.assertEqual({(tuple(r['macroblock']),r['lag']) for r in report['results']},
                             {((4,4),1),((4,4),3),((8,8),1),((8,8),3)})
            self.assertEqual(len(run.stdout.splitlines()),14)
            self.assertIn('4x4', run.stdout)
            self.assertIn('8x8', run.stdout)

    def test_fold_only_matches_delta_when_predictor_always_resets(self):
        plane = np.random.default_rng(4).integers(0,256,(8,8),dtype=np.uint8)
        report = analyze({'Y':plane}, {'Y':(1,1)}, {'width':8,'height':8},
                         [(2,2)], [1,3], (2,2))
        for raw, delta in zip(report['results'][::2],report['results'][1::2]):
            self.assertEqual(raw['fold_only']['total'], delta['total'])
            self.assertEqual(raw['fold_only']['channels']['Y']['by_lane_lsb_first'],
                             delta['channels']['Y']['by_lane_lsb_first'])

    def test_fold_uv_only_preserves_y_and_folds_chroma(self):
        rng = np.random.default_rng(8)
        planes = {'Y':rng.integers(0,256,(8,8),dtype=np.uint8),
                  'U':rng.integers(0,256,(4,4),dtype=np.uint8),
                  'V':rng.integers(0,256,(4,4),dtype=np.uint8)}
        scales = {'Y':(1,1),'U':(2,2),'V':(2,2)}
        report = analyze(planes,scales,{'width':8,'height':8},[(2,2)],[1,3],(4,4))
        for raw in report['results'][::2]:
            uv = raw['fold_uv_only']
            self.assertEqual(uv['channels']['Y']['by_lane_lsb_first'],raw['channels']['Y']['by_lane_lsb_first'])
            for name in ('U','V'):
                self.assertEqual(uv['channels'][name]['by_lane_lsb_first'],
                                 raw['fold_only']['channels'][name]['by_lane_lsb_first'])
            self.assertEqual(uv['total'],sum(c['total'] for c in uv['channels'].values()))
            baseline = raw['baseline_raw_raster_total']
            self.assertAlmostEqual(uv['reduction_vs_raw_percent'],100*(baseline-uv['total'])/baseline)
        gray = analyze({'Y':planes['Y']},{'Y':(1,1)}, {'width':8,'height':8},[(2,2)],[1])
        self.assertIsNone(gray['results'][0]['fold_uv_only'])
        self.assertIn('n/a',format_summary(gray))

    def test_native_jpeg_sampling(self):
        library = os.environ.get('TEST_TURBOJPEG_LIBRARY')
        if not library:
            self.skipTest('set TEST_TURBOJPEG_LIBRARY to test native JPEG decoding')
        with tempfile.TemporaryDirectory() as tmp:
            im = Image.fromarray(np.random.default_rng(2).integers(0,256,(9,13,3),dtype=np.uint8))
            for sampling, scale in ((0,(1,1)), (1,(2,1)), (2,(2,2))):
                path = Path(tmp)/f'{sampling}.jpg'
                im.save(path, subsampling=sampling)
                planes, scales, meta = load_image(path, turbojpeg_library=library)
                self.assertEqual(scales['U'], scale)
                self.assertEqual(planes['Y'].shape, (9,13))
                self.assertEqual(planes['U'].shape, ((9+scale[1]-1)//scale[1], (13+scale[0]-1)//scale[0]))
                result = analyze(planes, scales, meta, [scale], [1,2])
                self.assertEqual(len(result['results']), 12)
                if sampling == 2:
                    self.assertEqual(result['results'][0]['bus_bits'], 48)
                    with self.assertRaises(ValueError):
                        pack_chunks(planes, scales,13,9,(1,1))
            path = Path(tmp)/'rgb.png'
            im.save(path)
            planes, scales, meta = load_image(path)
            self.assertEqual(set(planes), set('RGB'))
            np.testing.assert_array_equal(planes['R'], np.asarray(im)[:,:,0])


if __name__ == '__main__':
    unittest.main()

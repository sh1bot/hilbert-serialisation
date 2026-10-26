import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from pixel_bus import coordinates, count_transitions, load_image, pack_chunks, analyze, transfer_layout, extend_lanes


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
        self.assertEqual(report['results'][3]['comparisons'], 21)
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
                self.assertEqual(len(result['results']), 6)
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

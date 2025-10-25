import HomographicMatcher as matcher
import VisualAnalyzer as visuals
import GroupingMetre as grouper
import HitsManager as hitsMngr
import Geometry2D as geo2D
import numpy as np
import cv2
import time
from datetime import datetime
import Scoreboard

class VideoAnalyzer:
    def __init__(self, videoPath, model, bullseye, ringsAmount, diamPx):
        '''
        {String} videoName - The path of the video to analyze
        {Numpy.array} model - An image of the target that appears in the video
        {Tuple} bullseye - (
                              {Number} x coordinate of the bull'seye location in the model image,
                              {Number} y coordinate of the bull'seye location in the model image
                           )
        {Number} ringsAmount - Amount of rings in the target
        {Number} diamPx - The diameter of the most inner ring in the target image [px]
        '''

        self.cap = cv2.VideoCapture(videoPath)
        _, test_sample = self.cap.read()
        frameSize = test_sample.shape
        self.rings_amount = ringsAmount
        self.inner_diam = diamPx
        self.model = model
        self.frame_h, self.frame_w, _ = frameSize
        if hasattr(cv2, 'SIFT_create'):
            self.sift = cv2.SIFT_create()
        else:
            self.sift = cv2.xfeatures2d.SIFT_create()

        # calculate anchor points and model features
        self.anchor_points, self.pad_model = geo2D.zero_pad_as(model, frameSize)
        self.pad_model_gray = cv2.cvtColor(self.pad_model, cv2.COLOR_RGB2GRAY)
        anchor_a = self.anchor_points[0]
        bullseye_anchor = (anchor_a[0] + bullseye[0],anchor_a[1] + bullseye[1])
        self.anchor_points.append(bullseye_anchor)
        self.anchor_points = np.float32(self.anchor_points).reshape(-1, 1, 2)
        self.model_keys, self.model_desc = self.sift.detectAndCompute(self.pad_model_gray, None)

        self.ring_radii_model = [self.inner_diam * (i + 1) for i in range(self.rings_amount)]
        pad_h, pad_w = self.pad_model_gray.shape
        yy, xx = np.indices((pad_h, pad_w), dtype=np.float32)
        bullseye_template = self.anchor_points[5][0]
        self.distance_template = np.sqrt(
            (xx - bullseye_template[0]) ** 2 + (yy - bullseye_template[1]) ** 2
        ).astype(np.float32)
        self.dt_scale = 0.5

        # Tracking state
        self.prev_gray = None
        self.tracks_prev = None  # previous frame tracked points
        self.H = None            # current homography from model->frame
        self.roi_poly = None     # last known quadrilateral of target (A,B,C,D)
        # Parameters
        self.lk_params = dict(winSize=(21,21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.min_tracks = 12
        self.seed_points = 60
        self.roi_margin = 0.12  # 12% expansion around last pose
        self._reset_telemetry()

    def _reset_telemetry(self):
        '''Reset timing telemetry for a new analysis run.'''
        self.telemetry_sections = {
            'target_detection': 0.0,
            'radial_filtering': 0.0,
            'radial_warp_subtract': 0.0,
            'radial_distance_map': 0.0,
            'radial_ring_estimate': 0.0,
            'radial_threshold_morph': 0.0,
            'radial_line_detect': 0.0,
            'radial_contour_reconstruct': 0.0,
            'hit_scoring': 0.0,
            'bookkeeping': 0.0,
            'bookkeeping_hits': 0.0,
            'bookkeeping_grouping': 0.0,
            'bookkeeping_overlay': 0.0,
            'bookkeeping_io': 0.0,
            'bookkeeping_idle': 0.0,
            'frame_total': 0.0,
            'run_clock': 0.0,
        }
        self.run_started_at = None
        self.run_finished_at = None
        self.total_runtime = 0.0
        self._run_perf_start = None

    def _roi_from_vertices(self, vertices):
        '''Create an expanded ROI polygon and mask from the given quadrilateral.'''
        # vertices: [A,B,C,D,E] where last is center; use first 4
        quad = np.array(vertices[:4], dtype=np.float32)
        # expand by margin using bounding box expansion for simplicity
        x_min = np.min(quad[:,0]); y_min = np.min(quad[:,1])
        x_max = np.max(quad[:,0]); y_max = np.max(quad[:,1])
        w = x_max - x_min; h = y_max - y_min
        x_min = max(0, int(x_min - self.roi_margin * w))
        y_min = max(0, int(y_min - self.roi_margin * h))
        x_max = min(self.frame_w-1, int(x_max + self.roi_margin * w))
        y_max = min(self.frame_h-1, int(y_max + self.roi_margin * h))
        box = np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]], dtype=np.int32)
        mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        cv2.fillPoly(mask, [box], 0xff)
        return box, mask

    def _seed_tracks(self, gray, roi_vertices):
        '''Seed a set of well-spaced points to track inside the ROI.'''
        _, mask = self._roi_from_vertices(roi_vertices)
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.seed_points, qualityLevel=0.01,
                                      minDistance=10, mask=mask, blockSize=7, useHarrisDetector=False)
        self.tracks_prev = pts

    def _reacquire(self, frame, prefer_roi=True, gray=None):
        '''Try to detect and lock a new homography using SIFT + BF; optionally restricted to ROI.'''
        mask = None
        if prefer_roi and self.roi_poly is not None:
            _, mask = self._roi_from_vertices(self.roi_poly)
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        matches, (train_keys, _) = matcher.ratio_match(self.sift, self.model_desc, gray, .7, mask)
        if len(matches) >= 4:
            Hcand = matcher.calc_homography(self.model_keys, train_keys, matches)
            if type(Hcand) != type(None):
                warped_transform = cv2.perspectiveTransform(self.anchor_points, Hcand)
                warped_vertices, warped_edges = geo2D.calc_vertices_and_edges(warped_transform)
                if matcher.is_true_homography(warped_vertices, warped_edges, (self.frame_w, self.frame_h), .2):
                    self.H = Hcand
                    self.roi_poly = warped_vertices
                    self._seed_tracks(gray, warped_vertices)
                    self.prev_gray = gray
                    return True, warped_vertices
        return False, None

    def _update_with_flow(self, prev_gray, gray):
        '''Update homography using LK-tracked points between frames. Returns (ok, vertices).'''
        if self.tracks_prev is None or len(self.tracks_prev) == 0:
            return False, None
        pts_prev = self.tracks_prev
        pts_curr, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_prev, None, **self.lk_params)
        if pts_curr is None or status is None:
            return False, None
        good_prev = pts_prev[status.reshape(-1) == 1]
        good_curr = pts_curr[status.reshape(-1) == 1]
        # reshape to Nx2 for motion estimation
        good_prev2 = good_prev.reshape(-1, 2)
        good_curr2 = good_curr.reshape(-1, 2)
        if len(good_curr2) < self.min_tracks:
            return False, None
        # Estimate incremental affine (similarity/translation scale+rot) for stability and speed
        A, inliers = cv2.estimateAffinePartial2D(good_prev2, good_curr2, method=cv2.RANSAC, ransacReprojThreshold=3.0, confidence=0.995)
        if A is None:
            return False, None
        # Convert 2x3 to 3x3 and compose with previous homography
        A33 = np.eye(3, dtype=np.float32)
        A33[:2,:3] = A
        self.H = A33 @ self.H
        # Update ROI and tracks
        warped_transform = cv2.perspectiveTransform(self.anchor_points, self.H)
        warped_vertices, _ = geo2D.calc_vertices_and_edges(warped_transform)
        self.roi_poly = warped_vertices
        # Keep current good points for next step
        self.tracks_prev = good_curr2.reshape(-1,1,2)
        return True, warped_vertices

    def _analyze_frame(self, frame):
        '''
        Analyze a single frame.

        Parameters:
            {Numpy.array} frame - The frame to analyze

        Returns:
            {Tuple} (
                        {Number} x coordinate of the bull'seye point in the target,
                        {Number} y coordinate of the bull'seye point in the target,
                    ),
            {list} [
                       {tuple} (
                                   {tuple} (
                                              {Number} x coordinates of the hit,
                                              {Number} y coordinates of the hit
                                           ),
                                   {Number} The hit's score according to the target's data
                               )
                       ...
                   ],
        '''

        # set default analysis meta-data
        scoreboard = []
        bullseye_point = None

        detect_start = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        warped_vertices = None
        # Decide whether to track or reacquire
        if self.H is None or self.prev_gray is None or self.tracks_prev is None:
            ok, warped_vertices = self._reacquire(frame, prefer_roi=True, gray=gray)
        else:
            ok, warped_vertices = self._update_with_flow(self.prev_gray, gray)
            if not ok:
                # try reacquire inside ROI first, then global
                ok, warped_vertices = self._reacquire(frame, prefer_roi=True, gray=gray)
                if not ok:
                    ok, warped_vertices = self._reacquire(frame, prefer_roi=False, gray=gray)

        self.telemetry_sections['target_detection'] += time.perf_counter() - detect_start

        if warped_vertices is not None:
            bullseye_point = warped_vertices[5]
            radial_total_start = time.perf_counter()

            warp_start = time.perf_counter()
            warped_img = cv2.warpPerspective(self.pad_model, self.H, (self.frame_w, self.frame_h))
            warped_transform = cv2.perspectiveTransform(self.anchor_points, self.H)
            _, warped_edges = geo2D.calc_vertices_and_edges(warped_transform)
            scale = geo2D.calc_model_scale(warped_edges, self.model.shape)
            sub_target = visuals.subtract_background(warped_img, frame)
            self.telemetry_sections['radial_warp_subtract'] += time.perf_counter() - warp_start

            estimated_warped_radius = self.rings_amount * self.inner_diam * scale[2]
            ring_radii = [r * scale[2] for r in self.ring_radii_model]
            circle_radius = ring_radii[-1]

            roi_margin = int(circle_radius * 0.15) + 8
            x0 = max(0, int(round(bullseye_point[0] - circle_radius - roi_margin)))
            x1 = min(self.frame_w, int(round(bullseye_point[0] + circle_radius + roi_margin)))
            y0 = max(0, int(round(bullseye_point[1] - circle_radius - roi_margin)))
            y1 = min(self.frame_h, int(round(bullseye_point[1] + circle_radius + roi_margin)))
            if x1 <= x0 or y1 <= y0:
                x0, y0, x1, y1 = 0, 0, self.frame_w, self.frame_h

            distance_start = time.perf_counter()
            warped_distances = cv2.warpPerspective(
                self.distance_template, self.H, (self.frame_w, self.frame_h))
            self.telemetry_sections['radial_distance_map'] += time.perf_counter() - distance_start
            pixel_distances_full = (None, warped_distances)
            dist_roi = warped_distances[y0:y1, x0:x1]

            sub_target_roi = sub_target[y0:y1, x0:x1]
            bullseye_roi = (bullseye_point[0] - x0, bullseye_point[1] - y0)

            proc_scale = getattr(self, 'dt_scale', 0.5)
            if proc_scale <= 0:
                proc_scale = 1.0
            if proc_scale != 1.0:
                proc_w = max(1, int(round((x1 - x0) * proc_scale)))
                proc_h = max(1, int(round((y1 - y0) * proc_scale)))
                sub_proc = cv2.resize(
                    sub_target_roi,
                    (proc_w, proc_h),
                    interpolation=cv2.INTER_AREA,
                )
                dist_proc = cv2.resize(
                    dist_roi,
                    (proc_w, proc_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                bullseye_proc = (bullseye_roi[0] * proc_scale, bullseye_roi[1] * proc_scale)
                estimated_radius_proc = estimated_warped_radius * proc_scale
                circle_radius_proc = circle_radius * proc_scale
            else:
                sub_proc = sub_target_roi
                dist_proc = dist_roi
                bullseye_proc = bullseye_roi
                estimated_radius_proc = estimated_warped_radius
                circle_radius_proc = circle_radius

            pixel_distances_proc = (None, dist_proc)
            circle_radius_proc, emphasized_proc = visuals.emphasize_lines(
                sub_proc,
                pixel_distances_proc,
                estimated_radius_proc,
                telemetry=self.telemetry_sections,
                ring_radius=circle_radius_proc,
            )

            contour_start = time.perf_counter()
            proj_contours_proc = visuals.reproduce_proj_contours(
                emphasized_proc, pixel_distances_proc, bullseye_proc, circle_radius_proc)
            self.telemetry_sections['radial_contour_reconstruct'] += time.perf_counter() - contour_start

            scale_back = (1.0 / proc_scale) if proc_scale != 0 else 1.0
            proj_contours = []
            for cont in proj_contours_proc:
                cont_scaled = cont.astype(np.float32) * scale_back
                cont_scaled[..., 0] += x0
                cont_scaled[..., 1] += y0
                proj_contours.append(cont_scaled.astype(np.int32))

            if not proj_contours:
                circle_radius_full, emphasized_full = visuals.emphasize_lines(
                    sub_target_roi,
                    (None, dist_roi),
                    estimated_warped_radius,
                    telemetry=None,
                    ring_radius=circle_radius,
                )
                fallback_contours = visuals.reproduce_proj_contours(
                    emphasized_full, (None, dist_roi), bullseye_roi, circle_radius)
                for cont in fallback_contours:
                    cont = cont.astype(np.float32)
                    cont[..., 0] += x0
                    cont[..., 1] += y0
                    proj_contours.append(cont.astype(np.int32))

            min_area = max(25.0, (circle_radius * 0.025) ** 2)
            filtered_contours = [c for c in proj_contours if cv2.contourArea(c) >= min_area]
            if filtered_contours:
                proj_contours = filtered_contours

            pixel_distances = pixel_distances_full

            self.telemetry_sections['radial_filtering'] += time.perf_counter() - radial_total_start

            scoring_start = time.perf_counter()
            suspect_hits = visuals.find_suspect_hits(proj_contours, warped_vertices, scale)

            max_radius = circle_radius + 5.0
            suspect_hits = [h for h in suspect_hits if h[2] <= max_radius]
            # calculate hits and draw circles around them
            scoreboard = hitsMngr.create_scoreboard(suspect_hits, scale, self.rings_amount, self.inner_diam)
            self.telemetry_sections['hit_scoring'] += time.perf_counter() - scoring_start

        # update previous gray for next iteration
        self.prev_gray = gray

        return bullseye_point, scoreboard

    def analyze(self, outputName, sketcher):
        '''
        Analyze a video completely and output the same video, with additional data written in it.

        Parameters:
            {String} outputName - The path of the output file
            {Sketcher} sketcher - A Sketcher object to use when writing the data to the output video
        '''

        self._reset_telemetry()
        hitsMngr.candidate_hits.clear()
        hitsMngr.verified_hits.clear()
        self.prev_gray = None
        self.tracks_prev = None
        self.H = None
        self.roi_poly = None
        self.run_started_at = datetime.now()
        self._run_perf_start = time.perf_counter()

        # set output configurations
        frame_size = (self.frame_w, self.frame_h)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(outputName, fourcc, 24.0, frame_size)

        while True:
            ret, frame = self.cap.read()

            if ret:
                frame_start = time.perf_counter()
                bullseye, scoreboard = self._analyze_frame(frame)
                bookkeeping_start = time.perf_counter()

                hits_start = time.perf_counter()
                for hit in scoreboard:
                    hitsMngr.sort_hit(hit, 30, 15)
                hitsMngr.discharge_hits()
                if bullseye is not None:
                    hitsMngr.shift_hits(bullseye)
                hits_elapsed = time.perf_counter() - hits_start
                self.telemetry_sections['bookkeeping_hits'] += hits_elapsed

                grouping_start = time.perf_counter()
                candidate_hits = hitsMngr.get_hits(hitsMngr.CANDIDATE)
                verified_hits = hitsMngr.get_hits(hitsMngr.VERIFIED)
                grouping_contour = grouper.create_group_polygon(frame, verified_hits)
                has_group = grouping_contour is not None
                grouping_diameter = grouper.measure_grouping_diameter(grouping_contour) if has_group else 0
                grouping_elapsed = time.perf_counter() - grouping_start
                self.telemetry_sections['bookkeeping_grouping'] += grouping_elapsed

                overlay_start = time.perf_counter()
                sketcher.draw_data_block(frame)
                verified_scores = [h.score for h in verified_hits]
                arrows_amount = len(verified_scores)
                sketcher.type_arrows_amount(frame, arrows_amount, (0x0,0x0,0xff))
                sketcher.type_total_score(frame, sum(verified_scores), arrows_amount * 10, (0x0,189,62))
                sketcher.type_grouping_diameter(frame, grouping_diameter, (0xff,133,14))
                sketcher.draw_grouping(frame, grouping_contour)
                sketcher.mark_hits(frame, candidate_hits, foreground=(0x0,0x0,0xff),
                                   diam=2, withOutline=False, withScore=False)
                sketcher.mark_hits(frame, verified_hits, foreground=(0x0,0xff,0x0),
                                   diam=5, withOutline=True, withScore=True)
                overlay_elapsed = time.perf_counter() - overlay_start
                self.telemetry_sections['bookkeeping_overlay'] += overlay_elapsed

                io_start = time.perf_counter()
                frame_resized = cv2.resize(frame, (1153, 648))
                cv2.imshow('Analysis', frame_resized)
                out.write(frame)
                key = cv2.waitKey(1) & 0xff
                io_elapsed = time.perf_counter() - io_start
                self.telemetry_sections['bookkeeping_io'] += io_elapsed

                frame_end = time.perf_counter()
                elapsed = frame_end - bookkeeping_start
                accounted = hits_elapsed + grouping_elapsed + overlay_elapsed + io_elapsed
                idle_elapsed = max(0.0, elapsed - accounted)
                self.telemetry_sections['bookkeeping_idle'] = (
                    self.telemetry_sections.get('bookkeeping_idle', 0.0) + idle_elapsed
                )
                self.telemetry_sections['bookkeeping'] += elapsed
                self.telemetry_sections['frame_total'] = self.telemetry_sections.get('frame_total', 0.0) + (frame_end - frame_start)
                if key == 27:
                    break
            else:
                print('Video stream is over.')
                break
                
        self.run_finished_at = datetime.now()
        if self._run_perf_start is not None:
            self.total_runtime = time.perf_counter() - self._run_perf_start
        self.telemetry_sections['run_clock'] = self.total_runtime

        verified_hits_final = list(hitsMngr.get_hits(hitsMngr.VERIFIED))
        scoreboard_path = Scoreboard.write_run_report(
            run_started_at=self.run_started_at,
            run_finished_at=self.run_finished_at,
            run_duration=self.total_runtime,
            telemetry=self.telemetry_sections,
            verified_hits=verified_hits_final,
            output_dir='res/output'
        )
        print(f'Scoreboard saved to {scoreboard_path}')

        # close window properly
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)



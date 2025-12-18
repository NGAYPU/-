import sys
import numpy as np
import cv2
import dlib

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

CANDIDATE_FILENAMES = {
    ord("1"): "leejaemyung.jpg",
    ord("2"): "kimmunsoo.jpg",
    ord("3"): "leejunseok.jpg",
}

detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
except Exception:
    print(f"ERROR: Failed to load Dlib model file: {PREDICTOR_PATH}")
    sys.exit(1)


def get_landmarks_and_rect(im):
    rects = detector(im, 1)
    if len(rects) == 0:
        return None, None

    rect = rects[0]
    landmarks = predictor(im, rect)
    return [(p.x, p.y) for p in landmarks.parts()], rect


def applyAffineTransform(src, srcTri, dstTri, size):
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(
        src,
        warpMat,
        (size[0], size[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return dst


def rectContains(rect, point):
    return (
        point[0] < rect[0] + rect[2]
        and point[0] > rect[0]
        and point[1] < rect[1] + rect[3]
        and point[1] > rect[1]
    )


def calculateDelaunayTriangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList()
    delaunayTri = []

    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if (
            rectContains(rect, pt1)
            and rectContains(rect, pt2)
            and rectContains(rect, pt3)
        ):
            ind = []

            for pt_t in [pt1, pt2, pt3]:
                for k in range(0, len(points)):
                    pt_p = points[k]
                    if abs(pt_t[0] - pt_p[0]) < 2 and abs(pt_t[1] - pt_p[1]) < 2:
                        ind.append(k)
                        break

            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

    return delaunayTri


def warpTriangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1Rect = []
    t2Rect = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2Rect), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = img1[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    img2Rect = img2Rect * mask

    roi = img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]]

    roi = roi * ((1.0, 1.0, 1.0) - mask)
    roi = roi + img2Rect

    img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = roi

    return img2


def load_candidate_data(filename):
    img = cv2.imread(filename)
    if img is None:
        return None, None

    points, rect = get_landmarks_and_rect(img)
    if points is None:
        return None, None

    return img, points


if __name__ == "__main__":

    candidate_data = {}
    for key, filename in CANDIDATE_FILENAMES.items():
        img, points = load_candidate_data(filename)
        if img is not None:
            candidate_data[key] = {"img": img, "points": points}

    if not candidate_data:
        print(
            "Error: No candidates loaded or face not detected in any candidate image. Exiting."
        )
        sys.exit(1)

    current_key = list(candidate_data.keys())[0]
    img_source = candidate_data[current_key]["img"]
    points1 = candidate_data[current_key]["points"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print(
                "Fatal Error: Could not open any webcam. Check connections/permissions."
            )
            sys.exit(1)

    while True:
        ret, img2 = cap.read()
        if not ret:
            break

        img2_float = np.float32(img2)
        img1Warped = np.copy(img2_float)

        points2, rect2_dlib = get_landmarks_and_rect(img2)

        if points2 is not None:

            hull2_indices = cv2.convexHull(np.array(points2), returnPoints=False)

            hull1 = [points1[int(i)] for i in hull2_indices]
            hull2 = [points2[int(i)] for i in hull2_indices]

            sizeImg2 = img2.shape
            rect = (0, 0, sizeImg2[1], sizeImg2[0])
            dt = calculateDelaunayTriangles(rect, hull2)

            if len(dt) > 0:
                for indices in dt:
                    t1 = [hull1[i] for i in indices]
                    t2 = [hull2[i] for i in indices]

                    img1Warped = warpTriangle(
                        np.float32(img_source), img1Warped, t1, t2
                    )

                hull8U = np.array(hull2, dtype=np.int32)
                mask = np.zeros(img2.shape, dtype=img2.dtype)
                cv2.fillConvexPoly(mask, hull8U, (255, 255, 255))

                r = cv2.boundingRect(np.float32([hull2]))
                center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))

                output = cv2.seamlessClone(
                    np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE
                )

                img2 = output

        cv2.imshow("Face Swapped Project (Press 1, 2, 3 to change, Q to quit)", img2)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key in CANDIDATE_FILENAMES:
            if key in candidate_data:
                img_source = candidate_data[key]["img"]
                points1 = candidate_data[key]["points"]

    cap.release()
    cv2.destroyAllWindows()

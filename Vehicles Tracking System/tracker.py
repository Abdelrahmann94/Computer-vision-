import math


class Tracker:
    def __init__(self):

        #save the center points of the detected objects
      self.center_points = {}

      self.id_count = 0

    def update(self, bounding_boxes):

        # Bounding boxes and ids
        objects_bb_ids = []

        # Get the center point of the object
        for box in bounding_boxes:
            x, y, w, h = box
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Check if that object has been alrealdy detected
            same_object_detected = False
            for id, pt in self.center_points.items():
                distance = math.hypot(cx - pt[0], cy - pt[1])

                if distance < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bb_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # Assign Id if the object wasn't detected before
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bb_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the center points dictionary to remove the IDs that are not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bb_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        
        self.center_points = new_center_points.copy()
        return objects_bb_ids
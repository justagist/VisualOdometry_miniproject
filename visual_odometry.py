import numpy as np
import cv2
import sys
import csv
from map_memory import Map_Memory
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class STAM():


    def __init__(self,scene,baseline_threshold = (100.0,80.0,35.0)):
        # Initialisation
        self.scene_no = scene
        self.camera_matrix, self.distortion_coeff = self.extract_intrinsics("S0%d_INPUT/intrinsicsS0%d.xml"%(self.scene_no,self.scene_no))
        self.scene_prefix = "S0%d_INPUT/S0%dL03_VGA/S0%dL03_VGA_"%(self.scene_no,self.scene_no,self.scene_no)
        self.patch_prefix = "S0%d_INPUT/S0%dL03_patch/S0%dL03_VGA_"%(self.scene_no,self.scene_no,self.scene_no)
        self.base_thresh = baseline_threshold[self.scene_no-1]
        self.patch_3d_file = "S0%d_INPUT/S0%d_3Ddata.csv"%(self.scene_no,self.scene_no)
        self.detect_alg = cv2.SIFT()
        self.matcher = cv2.BFMatcher()
        self.tot_repr_error = 0

    def match_features(self,img_gray, path_prefix,visualize_patch_flag=False):
        # uses the matchTemplate function to find the most likely match of the feature in the main image. The minMaxLoc function is used to find the most likely location of the feature in the image. 
        ## Returns an array of the most likey location for each of the features given in the first image.
        ## if visualize_patch_flag is set True, the patches will be marked and displayed in the first image. (Used for testing the function when there was only one image, and not a series of sequential images.)
        patch_no = 0
        patch_2d = np.array([],np.float64)
        present = True
        while present == True:
            patch = cv2.imread(path_prefix+"patch_%04d.png"%patch_no)
            if patch is None:
                if patch_no == 0:
                    sys.exit('ERROR! NO PATCHES FOUND!')
                else:
                    print "Total of %d patches obtained"%(patch_no)
                    present = False
            else:
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                w, h = patch.shape[::-1]
                result = cv2.matchTemplate(img_gray,patch,cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                patch_centre = ((max_loc[0]+max_loc[0]+w)/2,(max_loc[1]+max_loc[1]+h)/2)
                
                if patch_no == 0:
                    patch_2d = np.array([patch_centre],np.float64)
                else:
                    patch_2d = np.vstack((patch_2d,[patch_centre]))

                if visualize_patch_flag == True:
                    rect_start = max_loc
                    rect_end = (rect_start[0] + w, rect_start[1] + h)
                    cv2.rectangle(img_gray,rect_start, rect_end, 255, 2)
                    cv2.circle(img_gray, patch_centre, 2, 255, 3)
            
            patch_no += 1
        
        if visualize_patch_flag == True:
            cv2.imshow("Initial Patches",img_gray)
            cv2.waitKey(50) 
        
        return patch_2d
        
    def extract_intrinsics(self,xml_file):
        # Loading the Camera Matrix and the Distortion Coefficients
        loaded_data = cv2.cv.Load(xml_file)
        cam_mat = np.asarray(loaded_data)
        dist_mat = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        return cam_mat,dist_mat
            

    def test_results(self,pts_3d,pts_2d,P_mat):
        # Computes Reprojection Error in the Frame
        tot_error = 0
        for i in range(len(pts_2d)):

            pts_world_hom = np.array([[pts_3d[i][0]],[pts_3d[i][1]],[pts_3d[i][2]],[1]])
            pts_2d_projected_hom = np.dot(P_mat,pts_world_hom)
            pts_2d_projected = (pts_2d_projected_hom/pts_2d_projected_hom[2])[:2,0]
            tot_error+=np.linalg.norm(abs(pts_2d[i]-pts_2d_projected))

        self.tot_repr_error+=tot_error/len(pts_2d)
        print 'Mean Reprojection Error: %f'%(tot_error/len(pts_2d)),'\tNo. of Tracked Features:',len(pts_2d)
        self.mean_repr_error = tot_error/len(pts_2d)

    def test_projection_and_update(self):
        # Applying the `Reprojection Filter' Technique to remove correspondences with high reprojection error
        pt2d = self.mapmem.patch_2d_pts[len(self.mapmem.patch_2d_pts)-1]
        pt3d = self.mapmem.patch_3d_pts[len(self.mapmem.patch_3d_pts)-1]
        proj_mat = self.mapmem.proj_mats[len(self.mapmem.proj_mats)-1]
        del_idx = []
        for i in range(len(pt2d)):
            hom_wrld = np.array([[pt3d[i][0]],[pt3d[i][1]],[pt3d[i][2]],[1]])
            hom_prj = np.dot(proj_mat,hom_wrld)
            prj = (hom_prj/hom_prj[2])[:2,0]
            err = np.linalg.norm(abs(pt2d[i]-prj))
            if err > 25: #100 #25 for scene 1!,35 for scene 2  ## Threshold for the `Reprojection Filter'
                del_idx.append(i)
        self.mapmem.update_patches(pt2d,pt3d,'remove_from_last',del_idx)


    def get_projection_matrix(self,objectPoints, imagePoints):
        # Obtaining Projection Matrix and Camera Pose by solving the PnP problem using RANSAC
        rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, self.camera_matrix, self.distortion_coeff,reprojectionError=3)
        K = self.camera_matrix
        R = cv2.Rodrigues(rvec)[0]
        T = tvec
        trans_mat = np.concatenate((R,T),axis=1)
        proj_mat = np.dot(K,trans_mat)
        return proj_mat,trans_mat
        
    def get3Ddata(self,csv_file):
        # Get the world coordinates of the given initial features. Reading from file and storing as numpy array
        with open(csv_file, 'rb') as f:
            reader = csv.reader(f)
            x = []
            for row in reader:
                if len(x) == 0:
                    x = np.array([float(row[0]),float(row[1]),float(row[2])],np.float64)
                else:
                    x = np.vstack((x,[float(row[0]),float(row[1]),float(row[2])]))
        return x
           
    def check_baseline_and_update(self,ind):
        # Finds new feature correspondences if there is enough baseline
        if len(self.mapmem.cam_poses)>1 and len(self.mapmem.cam_poses) == ind+1:
            baseline_status = cv2.norm(self.mapmem.cam_poses[ind-1]-self.mapmem.cam_poses[ind]) >= self.base_thresh
            if baseline_status:
                new2d,new3d = self.find_new_patches(ind) # Calling the function to obtain new conrrespondences
                self.mapmem.update_patches(new2d,new3d,'add_modify',ind)
                new_proj,new_pose = self.get_projection_matrix(self.mapmem.patch_3d_pts[ind],self.mapmem.patch_2d_pts[ind])
                self.mapmem.store_proj_mat(new_proj,'replace',ind)
                self.mapmem.store_cam_pose(new_pose,'replace',ind)
                
        elif len(self.mapmem.cam_poses)>1 or ind>1:
            print 'Error: len(self.mapmem.pose_mats) = %d, index = %d'%(len(self.mapmem.pose_mats),ind)

    def find_new_patches(self,idx):
        # Computing SIFT descriptors for features in the the current and previous frames
        kp_curr,des_curr = self.detect_alg.detectAndCompute(self.mapmem.image_mats[idx],None)
        kp_prev,des_prev = self.detect_alg.detectAndCompute(self.mapmem.image_mats[idx-1],None)
        match_data_curr2prev = self.matcher.knnMatch(des_curr,des_prev, k=2)
        match_data_prev2curr = self.matcher.knnMatch(des_prev,des_curr, k=2)

        # Testing matches using Lowe's Ratio Test
        good_match_data_curr = self.ratio_test(match_data_curr2prev)
        good_match_data_prev = self.ratio_test(match_data_prev2curr)

        # Conducting Mutual Consistency Test for obtaining agreeing matches from the descriptors in the two images
        mutual_match_pts_curr,mutual_match_pts_prev = self.find_mutual_matches(good_match_data_curr,good_match_data_prev,kp_curr,kp_prev)

        # Using RANSAC to reject outliers
        new_2D_curr,new_2D_prev = self.find_Fmat_n_optimise_matches(mutual_match_pts_curr,mutual_match_pts_prev)

        # Correcting for distortions
        und_2D_curr,und_2D_prev = self.undistort_patches(new_2D_curr.reshape(-1,1,2),new_2D_prev.reshape(-1,1,2))
        new_3D_curr = self.triangulate_pts(und_2D_curr,und_2D_prev,idx)
        return und_2D_curr[:,0,:],new_3D_curr


    def ratio_test(self,matches):# ratio test as per Lowe's paper
        good = []
        i = 0
        for m,n in matches:
            if m.distance < 0.5*n.distance:#0.5 seems to be good for scene 1, 0.75 for scene 2
                good.append(matches[i])
            i += 1
        return good    

    def find_mutual_matches(self,mtch1,mtch2,kp1,kp2):
        # Finding mutually agreeing matches from the two sets of descriptors
        mutual_match = []
        k = 0
        for i in range(len(mtch1)):
            for j in range(len(mtch2)):
                if mtch1[i][0].queryIdx == mtch2[j][0].trainIdx and mtch2[j][0].queryIdx == mtch1[i][0].trainIdx:
                    # mutual_match.append(mtch1[i][0])
                    x2 = kp1[mtch1[i][0].queryIdx].pt[0]
                    y2 = kp1[mtch1[i][0].queryIdx].pt[1]
                    x1 = kp2[mtch1[i][0].trainIdx].pt[0]
                    y1 = kp2[mtch1[i][0].trainIdx].pt[1]
                    if k == 0:
                        pts1 = np.array([[x1,y1]],np.float64)
                        pts2 = np.array([[x2,y2]],np.float64)
                        k=1
                    else:
                        pts1 = np.vstack((pts1,[[x1,y1]]))
                        pts2 = np.vstack((pts2,[[x2,y2]]))
        return pts1,pts2

    def find_Fmat_n_optimise_matches(self,pts1,pts2):
        # Applying RANSAC to remove outliers. Inlier threshold = 1 pixel.
        F_mat, inliers = cv2.findFundamentalMat(np.float32(pts2),np.float32(pts1),method=cv2.cv.CV_FM_RANSAC,param1 = 1.0,param2=0.99)
        j = 0
        for ins in range(len(inliers)):
            if inliers[ins] == 1:
                if j == 0:
                    new2d_1 = np.array([pts1[ins]],np.float64)
                    new2d_2 = np.array([pts2[ins]],np.float64)
                    j = 1
                else:
                    new2d_1 = np.vstack((new2d_1,[pts1[ins]]))
                    new2d_2 = np.vstack((new2d_2,[pts2[ins]]))
        return new2d_1,new2d_2

    def undistort_patches(self,p1,p2): # Correcting for distortion
        und1 = cv2.undistortPoints(p1, self.camera_matrix, self.distortion_coeff,P=self.camera_matrix)
        und2 = cv2.undistortPoints(p2, self.camera_matrix, self.distortion_coeff,P=self.camera_matrix)
        return und1,und2

    def triangulate_pts(self,p_curr,p_prev,indx):
        # Finding the 3D location of a point using its coordinates in the two images
        self.mapmem.proj_mats[indx]
        hom_curr_4D = cv2.triangulatePoints(self.mapmem.proj_mats[indx-1], self.mapmem.proj_mats[indx], p_prev, p_curr)
        curr_3D = hom_curr_4D/hom_curr_4D[3]
        curr_3D =  curr_3D.T
        return curr_3D[:,:3]

    def updatepatch_klt(self,prev_img,curr_img,prev_2d):
        # Using Lucas-Kanade Optical Flow method to track features across Images. Gives the image location of the features from the previous frame in the current frame
        patch, patch_status, err = cv2.calcOpticalFlowPyrLK(prev_img,curr_img,np.float32(prev_2d))
        j = 0
        lost_point_indices = []
        for i in range(len(patch)):
            if patch_status[i] == 1 and err[i] < 10: #best result: 10 for scenes 1 & 2
                if j == 0:
                    new_patch = np.array([patch[i]],np.float64)
                    j = 1
                else:
                    new_patch = np.vstack((new_patch,[patch[i]]))
            else:
                lost_point_indices.append(i)
        return new_patch,lost_point_indices

    def process(self,index):
        imgs = cv2.imread(self.scene_prefix+"%04d.png"%index) # Read image and convert to grayscale
        gray_img = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
        self.mapmem.store_image_mats(gray_img)
        if index == 0:
            patch_3D = self.get3Ddata(self.patch_3d_file) # Read 3D data of features from file
            patch_2D = self.match_features(gray_img,self.patch_prefix) # Find the initial templates in the first frame
            self.mapmem.update_patches(patch_2D,patch_3D)
        else:
            patch_2D,lost_pt_index = self.updatepatch_klt(self.mapmem.image_mats[index-1],gray_img,self.mapmem.patch_2d_pts[index-1]) # Track features from the previous frame using Optical Flow
            self.mapmem.update_patches(patch_2D,self.mapmem.patch_3d_pts[index-1],'remove_lost',lost_pt_index) # Remove features for which optical flow matches were not found
            patch_3D = self.mapmem.patch_3d_pts[index]
        projection_matrix,cam_pose = self.get_projection_matrix(patch_3D, patch_2D) # Estimate projection matrix and camera pose
        self.mapmem.store_proj_mat(projection_matrix)
        self.mapmem.store_cam_pose(cam_pose)
        return imgs

    def visualize_cam_pose(self,pose_mat):
        # Plotting the estimated camera trajectory
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        poses = np.asarray(self.mapmem.cam_poses).reshape(-1,3)
        z = poses[:,1]
        y = poses[:,0]
        x = poses[:,2]
        ax.scatter(x, y, z, c='r', marker='o')
        ax.scatter(x[0],y[0],z[0],c='g',marker='x',s=500)
        ax.legend()
        ax.set_xlabel('z')
        ax.set_ylabel('x')
        ax.set_zlabel('y')
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-200, 2000)
        ax.set_zlim(-1000, 0)
        ax.invert_zaxis()
        ax.invert_xaxis()
        
    def start(self):
        self.mapmem = Map_Memory() # Class for storing all the data: image frames, 2D points, 3D points, projection matrices, camera poses etc.
        i = 0
        self.run_status = True
        while not cv2.imread(self.scene_prefix+"%04d.png"%i) is None and self.run_status == True:
            curr_img = self.process(i)

            # If enough features are not being tracked, check baseline and find new features
            if len(self.mapmem.patch_2d_pts[i])<200:
                self.check_baseline_and_update(i)

            # -------- `Reprojection Filter' -------------- (use for scenes 1 and 2 only)
            self.test_projection_and_update()
            
            # Indicating features using markers in the image
            for point in self.mapmem.patch_2d_pts[i]:
                cv2.circle(curr_img, (int(point[0]),int(point[1])), 1, (0,0,255), 3)

            cv2.imshow("Tracked Points",curr_img) # Visualising the tracking of features
            cv2.moveWindow("Tracked Points", 1, 1)
            cv2.waitKey(1)
            print i,
            # Calculating mean reprojection error of the frame
            self.test_results(self.mapmem.patch_3d_pts[i],self.mapmem.patch_2d_pts[i],self.mapmem.proj_mats[i])
            i+=1
        
        if i != 0:
            print i,'total mean error:',self.tot_repr_error/i

            # Plotting the estimated camera trajectory
            self.visualize_cam_pose(self.mapmem.cam_poses[len(self.mapmem.cam_poses)-1])
            plt.show()
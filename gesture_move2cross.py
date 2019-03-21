import cv2
import scipy.io as sio
from statistics import *
from pylab import *
import numpy as np
from skimage import measure,morphology,exposure
import matplotlib.pyplot as plt

def skindetect2(RGB):
    R,G,B = RGB[:,:,0],RGB[:,:,1],RGB[:,:,2]
    #   手动颜色空间转换
    Y = (0.2568*np.array(R) + 0.5041*np.array(G) + 0.0979*np.array(B)+16).astype(uint8)
    Cb = (0.4392 * np.array(B)-0.1482 * np.array(R) -0.2910 * np.array(G) + 128).astype(uint8)
    Cr = (0.4392 * np.array(R) -0.3678 * np.array(G) -0.0714 * np.array(B) + 128).astype(uint8)
    I = RGB
    rows,columns = Y.shape
    k = (2.53/180)*math.pi
    m = math.sin(k)
    n = math.cos(k)
    x,y = 0,0
    cx,cy,ecx,ecy,a,b = 109.38,152.02,1.60,2.41,25.39,14.03
    for i in range(rows):
        for j in range(columns):
            if Y[i,j] < 80:
                I[i,j,:] = 0
            elif Y[i,j] <= 230 and Y[i,j] >= 80:
                x=(double(Cb[i,j])-cx)*n+(double(Cr[i,j])-cy)*m
                y=(double(Cr[i,j])-cy)*n-(double(Cb[i,j])-cx)*m
                if (((x-ecx)**2)/a**2+(y-ecy)**2/b**2) <= 1:
                    I[i,j,:] = 255
                else:
                    I[i,j,:] = 0
            elif Y[i,j] > 230:
                x=(double(Cb[i,j])-cx)*n+(double(Cr[i,j])-cy)*m
                y=(double(Cr[i,j])-cy)*n-(double(Cb[i,j])-cx)*m
                if ((x-ecx)**2/(1.1*a)**2+(y-ecy)**2/(1.1*b)**2) <= 1:
                    I[i,j,:] = 255
                else:
                    I[i,j,:] = 0
    I = cv2.cvtColor(I,cv2.COLOR_RGB2GRAY)
    #   MATLAB默认二值化的阈值是0.5*255，此处设置为120是根据结果来调整的，后面的1是设置了范围，为了与MATLAB结果一致
    retval,I = cv2.threshold(I,0.5*255,1, cv2.THRESH_BINARY)
    return I

def autoroi2(bwimage):
    row,col = np.where(bwimage==1)
    col_min = np.min(col)
    col_max = np.max(col)
    row_min = np.min(row)
    row_max = np.max(row)
    y_roi = col_min
    x_roi = row_min
    try:
        width_roi = col_max - col_min
    except ValueError:
        width_roi = 0
    try:
        height_roi = row_max - row_min
    except ValueError:
        height_roi = 0
    if len(row) == 0:
        hand_roi = bwimage
        x = 0
        y = 0
        w = 0
        h = 0
    else:
        hand_roi = bwimage[x_roi:x_roi + height_roi,y_roi:y_roi + width_roi]
        w = height_roi
        h = width_roi
        x = y_roi
        y = x_roi
    return hand_roi,x,y,w,h

def Gcenter(I):
	I = np.array(I).astype(np.uint8)
	rows,cols = I.shape
	x = np.dot(np.ones((rows,1)),np.arange(1,cols+1).reshape(1,cols)).astype(np.uint8)
	y = np.dot(np.arange(1,rows+1).T.reshape(rows,1),np.ones((1,cols))).astype(np.uint8)
	area = np.sum(I == 1)
	meanx = np.sum(I*x)/area
	meany = np.sum(I*y)/area
	return meanx,meany

def ShowResult(result):
    if result == 1:
        word = 'yes'
    elif result == 2:
        word = 'cannot judge'
    elif result == 3:
        word = 'no(done)'
    elif result == 4:
        word = 'yes(ing)'
    elif result == 5:
        word = 'yes(done)'
    else:
        word = 'no'
    return word

def illumination_correct(im):
	HSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV) / 255
	V = HSV[:, :, 2]
	HSIZE = min(im.shape[0], im.shape[1])
	# 卷积核的大小必须为奇数
	if HSIZE % 2 == 0:
		HSIZE -= 1
	q = sqrt(2)
	SIGMA1 = 15
	SIGMA2 = 80
	SIGMA3 = 250
	gaus1 = cv2.GaussianBlur(V, (HSIZE, HSIZE), SIGMA1 / q)
	gaus2 = cv2.GaussianBlur(V, (HSIZE, HSIZE), SIGMA2 / q)
	gaus3 = cv2.GaussianBlur(V, (HSIZE, HSIZE), SIGMA3 / q)
	gaus = (gaus1 + gaus2 + gaus3) / 3
	m = np.mean(gaus)
	gama = np.power(0.5, (m - gaus) / m)
	V = np.power(V, gama)
	HSV[:, :, 2] = V
	HSV = uint8(HSV * 255)
	rgb = cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB)
	return rgb

def ColorEnhance(BGR):
	HSV = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV) / 255
	V = HSV[:,:,2]
	V = exposure.rescale_intensity(V,in_range=(0.2,0.8),out_range=(0,1))
	V = exposure.adjust_gamma(V,0.6)
	HSV[:,:,2] = V
	HSV = uint8(HSV*255)
	HSVrgb = cv2.cvtColor(HSV,cv2.COLOR_HSV2RGB)
	return HSVrgb

def Comp_foam_new(IM,bwt,x,y):
    x_point = int(np.round(x[0]))
    y_point = int(np.round(y[0]))
    width =  int(y[1]-y[0]) + 1
    height = int(x[1]-x[0]) + 1
    box = IM[x_point:x_point+height,y_point:y_point+width].copy()
    box_small = cv2.resize(box,(math.ceil(width*0.3),math.ceil(height*0.3)),interpolation=cv2.INTER_NEAREST)
    box_c = illumination_correct(box_small)
    I = cv2.cvtColor(box_c,cv2.COLOR_RGB2GRAY)
    retval, bw = cv2.threshold(I, bwt * 255, 1, cv2.THRESH_BINARY)
    s = np.sum(bw == 1)/bw.size
    return s,bw,I

def Comp_foam_gs(box,bwt):
	box_ic = illumination_correct(box)
	I = cv2.cvtColor(box_ic,cv2.COLOR_RGB2GRAY)
	ret, bw = cv2.threshold(I,bwt*255,1,cv2.THRESH_BINARY)
	s = np.sum(bw==1)/bw.size
	return s,bw

def Comp_effluent(IM,I_t,thresh,x,y):
	I_t = cv2.cvtColor(I_t, cv2.COLOR_BGR2GRAY)
	x_point = int(x[0])
	y_point = int(y[0])
	width = int(y[1] - y[0])+1
	height = int(x[1] - x[0])+1
	I_tc = I_t[x_point:x_point + height, y_point:y_point + width].copy()
	box = IM[x_point:x_point + height, y_point:y_point + width].copy()
	box = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
	diff = double((abs(double(box) - double(I_tc))))
	retval, bw = cv2.threshold(diff, thresh * 255, 1, cv2.THRESH_BINARY)
	bw = np.array(bw)
	score = np.sum(bw == 1) / bw.size
	return score

def Comp_soap_new(IM,x,y):
	x_point = int(x[0])
	y_point = int(y[0])
	width = int(y[1] - y[0])+1
	height = int(x[1] - x[0])+1
	box = IM[x_point:x_point + height, y_point:y_point + width].copy()
	box = illumination_correct(box)
	hand = skindetect2(box)
	hll = np.sum(hand == 1) / hand.size
	return hll,hand

def Comp_wash(IM,x,y):
	x_point = int(x[0])
	y_point = int(y[0])
	width = int(y[1] - y[0]) + 1
	height = int(x[1] - x[0]) + 1
	box = IM[x_point:x_point + height, y_point:y_point + width].copy()
	box = cv2.resize(box, (math.ceil(width * 0.3), math.ceil(height * 0.3)), interpolation=cv2.INTER_NEAREST)
	box = illumination_correct(box)
	hand = skindetect2(box)
	return hand

def Judge_effluent(score,result_total,j,mint,maxt,ef_et):
	if score > mint and score < maxt:
		result = 1
	elif score >= maxt:
		try:
			if j <= ef_et:
				result = mode(result_total[:j])
			else:
				result = mode(result_total[(j - ef_et - 1):(j - 1)])
		except StatisticsError:
			result = 0
	else:
		result = 0
	return result

def Judge_foam(num,j,ef,crnt,pfnt,tpft):
    if j<=ef:
        pfnt[j-1] = sum(num[:j]>crnt)
    else:
        pfnt[j-1] = sum(num[j-ef-1:j]>crnt)
    length = 0
    for i in range(len(pfnt)):
        for j in range(len(pfnt[0])):
            if pfnt[i][j] > pfnt:
                length += 1
    if length > tpft:
        result = 1
    else:
        result = 0
    return result

def Judge_soap(c_sp,j,fps,tt):
    if c_sp[j] ==0:
        if (len([i for i in c_sp[:j] if i ==1])) / fps <tt:
            result = 0
        else:
            result = 3
    else:
        if (len([i for i in c_sp[:j] if i ==1])) / fps <tt:
            result=4
        else:
            result=5
    return result

def Judge_wash(c,j,ef):
    if j <= ef:
        try:
            if mode(c[:j]) == 1:
                result = 1
            else:
                result = 0
        except StatisticsError:
            return 0
    else:
        if mode(c[j-ef-1:j]) == 1:
            result = 1
        else:
            result = 0
    return result

def Judge_washcurrent(hand,result):
    col = []
    for i in range(len(hand)):
        for j in range(len(hand[0])):
            if hand[i][j] == 1:
                col.append(j)
    if result == 1:
        if min(col,default=0)<(0.2*len(hand[0])):
            c = 1
        else:
            c = 0
    else:
         c = 0
    return c

if __name__ == '__main__':
	video_full_path = "Video/yy3.mp4"
	capture = cv2.VideoCapture(video_full_path)

	# opencv2.4.9用cv2.cv.CV_CAP_PROP_FPS；如果是3用cv2.CAP_PROP_FPS
	(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
	if int(major_ver) < 3:
		fps = capture.get(cv2.cv.CV_CAP_PROP_FPS)
	else:
		fps = capture.get(cv2.CAP_PROP_FPS)

	# 获取所有帧
	frame_count = 0
	all_frames = {}
	while True:
		ret, frame = capture.read()
		if not ret:
			break
		all_frames[str(frame_count)] = frame
		frame_count = frame_count + 1

	data = sio.loadmat('ROI_yy3-4.mat')
	x_et = data['x_et']
	x_fm = data['x_fm']
	x_gs = data['x_gs']
	x_sp = data['x_sp']
	x_ws = data['x_ws']
	y_et = data['y_et']
	y_fm = data['y_fm']
	y_gs = data['y_gs']
	y_sp = data['y_sp']
	y_ws = data['y_ws']

	# I_t为第一帧也就是背景
	I_t = all_frames["0"]

	# Parameters
	min_et = 0.25
	max_et = 0.45
	bwt_et = 30/255
	ef_et = 5

	bwt_ws = 50/255
	dlt_ws = 10
	ef_ws = 20

	lr_sp = 0.1
	tt_sp = 0.5
	lt_sp = 100

	ef_fm = np.ceil(fps)*2
	bwt_fm = 200/255
	dlt_fm = 15
	rrt_fm = 0.12 #yy3:0.12  yy5:0.07
	prtt_fm = 15
	tpft_fm = 15

	mina_st = 100
	maxa_st = 700

	ef_it = 2
	ef_jy = 5
	jyt_mv = 10
	jyt_it = 10
	pft_it = 60
	rgt_it = 100
	stt_it = 1
	rt_it = 0.6

	snt_cs =20
	dlr_it = 0.33
	strelr_cs = 0.086
	tt_cs = 5
	crnt_cs = 4
	pfnt_cs = 25

	tt_bk = 1
	at_bk =1200
	pfnt_bk = 20
	tt_lrbk = 1
	dbt_lrbk = 10
	pfnt_lrbk = 6.9

	stf = 100
	ovf = frame_count
	frame_num = len(all_frames)
	step1 = True

	# 预先分配内存
	crn_mv = np.arange(frame_num)*0.0
	pfn_st = np.arange(frame_num)*0.0
	set = np.arange(frame_num)*0.0
	wi = np.arange(frame_num)*0.0
	dw = np.arange(frame_num)*0.0
	jy = np.arange(frame_num)*0.0
	rst_it = np.arange(frame_num)*0.0
	crn_cs = np.arange(frame_num)*0.0
	pfn_cs = np.arange(frame_num)*0.0
	rst_cs = np.arange(frame_num)*0.0
	crn_bk = np.arange(frame_num)*0.0
	area2 = np.arange(frame_num)*0.0
	bottom = np.arange(frame_num)*0.0
	db = np.arange(frame_num)*0.0
	darea = np.arange(frame_num)*0.0
	pfn_bk = np.arange(frame_num)*0.0
	rst_bk = np.arange(frame_num)*0.0
	pfn_lrbk = np.arange(frame_num)*0.0
	rst_lrbk = np.arange(frame_num)*0.0
	rst_rbk = np.arange(frame_num)*0.0
	rst_lbk = np.arange(frame_num)*0.0
	s_et = np.arange(frame_num)*0.0
	result_et = np.arange(frame_num)*0.0
	c_w = np.arange(frame_num)*0.0
	result_ws = np.arange(frame_num)*0.0
	c_sp = np.arange(frame_num)*0.0
	result_sp =np.arange(frame_num)*0.0
	rt_fm = np.arange(frame_num)*0.0
	prt_fm = np.arange(frame_num)*0.0
	rst_fm = np.arange(frame_num)*0.0
	result_fm = np.arange(frame_num)*0.0

	# ROI region
	x_point_gs = int(x_gs[0])
	y_point_gs = int(y_gs[0])
	width_gs = int(y_gs[1] - y_gs[0])+1
	height_gs = int(x_gs[1] - x_gs[0])+1

	# Precomputation
	fps = int(fps)
	ef_cs = fps * tt_cs
	time_start = time.time()
	_,bw_It,_ = Comp_foam_new(I_t,bwt_fm,x_gs,y_gs)

	# Process each frame
	for i in range(stf,frame_num):
		IM = all_frames[str(i)]
		j = i-stf

		# effluent judging
		s_et[j] = Comp_effluent(IM,I_t,bwt_et,x_et,y_et)
		result_et[j] = Judge_effluent(s_et[j],result_et,j,min_et,max_et,ef_et)

		# hand washing judging
		hand_ws = Comp_wash(IM,x_ws,y_ws)
		c_w[j] = Judge_washcurrent(hand_ws,result_et[j])
		result_ws[j] = Judge_wash(c_w,j,ef_ws)
		time_ws = np.sum(result_ws==1)/fps

		# soap gathering judging
		hl,hand_sp = Comp_soap_new(IM,x_sp,y_sp)
		if hl>lr_sp:
			c_sp[j] = 1
		else:
			c_sp[j] = 0
		result_sp[j] = Judge_soap(c_sp,j,fps,tt_sp)

		# foam generating judging
		if result_sp[j]==3:
			rt_fm[j],bw_fm,gray = Comp_foam_new(IM,bwt_fm,x_fm,y_fm)
			if j<=ef_fm-1:
				prt_fm[j] = np.sum(rt_fm[:j]>rrt_fm)
			else:
				prt_fm[j] = np.sum(rt_fm[int(j-ef_fm-1):j]>rrt_fm)

			if np.sum(prt_fm>prtt_fm)>tpft_fm:
				rst_fm[j] = 1
			else:
				rst_fm[j] = 0
			if np.sum(rst_fm==1)>1:
				result_fm[j] = 1
			else:
				result_fm[j] = 0
		else:
			result_fm[j] = 0

		if result_sp[j] ==3:
			step1 = False
			box = IM[x_point_gs:x_point_gs + height_gs,y_point_gs:y_point_gs + width_gs]
			box = cv2.resize(box, (math.ceil(width_gs * 0.3), math.ceil(height_gs * 0.3)), interpolation=cv2.INTER_NEAREST)
			box_c = ColorEnhance(box)
			I = skindetect2(box_c)
			I = I.astype(np.bool)

			#***********************************  Palm  *********************************#
			rgdl = morphology.remove_small_objects(I, min_size=5, connectivity=2)
			_, xi, yi, wi[j], hi = autoroi2(rgdl)
			Ifill = morphology.binary_closing(rgdl)
			Ifill[:,0] = 0

			InverseIfill = (Ifill == 0)
			#	将图片的最后一列的颜色改变，否则无法检测出两手交叉的连通域
			InverseIfill[:,InverseIfill.shape[1]-1] = 0
			#	标记出各个连通域，MATLAB默认8邻域，所以connectivity=2
			Label_InverseIfill = measure.label(InverseIfill,connectivity=2)
			#	获取各个连通域的属性，只有标记经过上面的步骤后才能获取所有，否则只获取整张图片属性
			I_reg_mv= measure.regionprops(Label_InverseIfill)
			Area_mv = []
			for index in range(len(I_reg_mv)):
				Area_mv.append([I_reg_mv[index].area,index])
			Area_mv = sorted(Area_mv,reverse=True)
			crn_mv[j] = len(I_reg_mv)

			if crn_mv[j]>1:
				if Area_mv[1][0]>mina_st and Area_mv[1][0]<maxa_st:
					pfn_st[j] = 1
				else:
					Area_mv[1][0] = 0
					pfn_st[j] = 0

			if j<=stt_it*np.ceil(fps):
				if np.sum(pfn_st==1)/(stt_it*np.ceil(fps))>rt_it:
					set[j] = 1
				else:
					set[j] = 0
			else:
				if np.sum(pfn_st[int(j-stt_it*np.ceil(fps)):]==1)/(stt_it*np.ceil(fps))>rt_it:
					set[j] = 1
				else:
					set[j] = 0

			if np.sum(set==1)>0:
				if crn_mv[j]>1:
					if Area_mv[1][0]> rgt_it:
						y_end =I_reg_mv[Area_mv[1][1]].bbox[1]
						wi[j] = abs(y_end-yi)

					if j<=0:
						dw[j] = 0
					else:
						dw[j] = wi[j] - wi[j-1]

					if j<ef_jy-1:
						jy[j] = np.sum(np.abs(dw[0:j]))
					else:
						jy[j] = np.sum(np.abs(dw[j-ef_jy:j]))
				else:
					jy[j] = 0

				if jy[j]>jyt_mv:
					c_it = 1
				else:
					c_it = 0

				if np.sum(jy>jyt_it)>pft_it:
					rst_it[j] = 1
				else:
					rst_it[j] = 0
			else:
				rst_it[j] = 0
				c_it = 0

			if sum(rst_it==1)>0:
				result_it = 1
			else:
				result_it = 0

			#********************************* Cross ******************************#
			if np.sum(set==1)>snt_cs:
				hw = np.min(wi[j-10:j])
			else:
				hw = box_c.shape[0]/2

			Ifill2 = morphology.closing(I)
			Ifill3 = Ifill2
			meanx,meany = Gcenter(Ifill2)
			strel_para = int(round(strelr_cs * hw))
			sel = morphology.disk(strel_para)
			Ifill2 = np.array(Ifill2).astype(np.uint8)
			Io = morphology.opening(Ifill2, sel)
			# Io = morphology.closing(I, sel)
			Iod = (Io==0)
			Iod[:, int(round(meany)):] = 0
			bwdiff = Ifill2*Iod

			dlt_para = np.ceil(dlr_it*hw)
			finger = morphology.remove_small_objects(bwdiff, min_size=dlt_para, connectivity=2)

			Label_finger = measure.label(finger, connectivity=2)
			Finger_reg = measure.regionprops(Label_finger)
			crn_cs[j] = len([i for i in Finger_reg[:len(Finger_reg)] if i['area'] > 18])

			if j<=ef_cs-1:
				pfn_cs[j] = np.sum(crn_cs[:j]>=crnt_cs)
			else:
				pfn_cs[j] = np.sum(crn_cs[j-ef_cs:j]>=crnt_cs)

			if c_it == 1:
				if pfn_cs[j]>pfnt_cs:
					rst_cs[j] = 1
				else:
					rst_cs[j] = 0
			else:
				rst_cs[j]=0

			if np.sum(rst_cs==1)>1:
				result_cs = 1
			else:
				result_cs = 0

			#********************************** Back *******************************#
			_, bw_fm = Comp_foam_gs(box, bwt_fm)

			bw_fm2 = bw_fm*(bw_It==0)
			orimg = np.bitwise_or(bw_fm2,Ifill3)
			orimg = np.array(orimg,dtype=bool)
			dlorextr = morphology.remove_small_objects(orimg,min_size=15,connectivity=2)

			sel = morphology.disk(3)
			Ic1 = morphology.closing(dlorextr,sel)
			Ic1[:,0] = 0

			InverseIc1 = (Ic1==0)

			# cv2.imshow("1",np.uint8(InverseIc1*255))
			Label_Ic1 = measure.label(InverseIc1,connectivity=2)
			I_reg_bk = measure.regionprops(Label_Ic1)
			crn_bk[j] = len(I_reg_bk)

			if np.sum(set==1)>0:
				Area= []
				for idx in range(len(I_reg_bk)):
					Area.append([I_reg_bk[idx].area,idx])
				Area = sorted(Area,reverse=True)
				if crn_bk[j]>1:
					if Area[1][0]>rgt_it:
						area2[j] = Area[1][0]
						bottom[j] = I_reg_bk[Area[1][1]].bbox[2]
					else:
						area2[j] = 0
						bottom[j] = 0
			else:
				area2[j] = 0
				bottom[j] = 0

			if j<2 or area2[j]==0 or area2[j-2]==0:
				db[j] = 0
			else:
				db[j] = bottom[j] - bottom[j-2]

			#******************** 判断面积差 *********************#
			if j<=tt_bk*np.ceil(fps):
				if len(area2[area2!=0])==0:
					darea[j] = 0
				else:
					darea[j] = np.max(area2) - np.min(area2[area2!=0])
				pfn_bk[j] = np.sum(darea>at_bk)
			else:
				prarea = area2[int(j-tt_bk*np.ceil(fps)):]
				if len(prarea[prarea!=0])==0:
					darea[j] = 0
				else:
					darea[j] = np.max(prarea)-np.min(prarea[prarea!=0])
				pfn_bk[j] = np.sum(darea[int(j-tt_bk*np.ceil(fps)):]>at_bk)

			if c_it==1:
				if pfn_bk[j]>pfnt_bk:
					rst_bk[j] = 1
				else:
					rst_bk[j] = 0
			else:
				rst_bk[j] = 0

			#**************************** 判断左右手部分 *************************#
			if j<=tt_lrbk*np.ceil(fps):
				pfn_lrbk[j] = np.sum(db>dbt_lrbk)
			else:
				pfn_lrbk[j] = np.sum(db[int(j-tt_lrbk*np.ceil(fps)):j]>dbt_lrbk)

			if c_it==1:
				if pfn_lrbk[j]>pfnt_lrbk:
					rst_lrbk[j] = 1
				else:
					rst_lrbk[j] = 0

				if rst_bk[j]==1:
					if rst_lrbk[j]==1:
						rst_rbk[j] = 1
					else:
						rst_lbk[j] = 1
				else:
					rst_lbk[j] = 0
					rst_rbk[j] = 0
			else:
				rst_lbk[j] = 0
				rst_rbk[j] = 0

			if np.sum(rst_lbk==1)>1:
				result_lbk = 1
			else:
				result_lbk = 0
			if np.sum(rst_rbk==1)>1:
				result_rbk = 1
			else:
				result_rbk = 0
		else:
			c_it = 0
			result_it = 0
			result_cs = 0
			result_lbk = 0
			result_rbk = 0

		# show the result
		word_et = ShowResult(result_et[j])
		word_ws = ShowResult(result_ws[j])
		word_sp = ShowResult(result_sp[j])
		word_fm = ShowResult(result_fm[j])
		word_cit = ShowResult(c_it)
		word_cs = ShowResult(result_cs)
		word_it = ShowResult(result_it)
		word_lbk = ShowResult(result_lbk)
		word_rbk = ShowResult(result_rbk)
		font = cv2.FONT_HERSHEY_SIMPLEX

		cv2.putText(IM,str(i),(5,25),font,0.7,(255,0,255),2)
		cv2.putText(IM,'Water:%s'%str(word_et)+' '+'Wash:%s'%str(word_ws)+' '+'Soap:%s'%str(word_sp)+' '+'Foam:%s'%str(word_fm),
					(5,65),font,0.7,(255,0,255),2)
		cv2.putText(IM, 'Palm:%s' % (str(word_it))
					 + ' ' + 'Cross:%s'%str(word_cs)+' '+"Left:%s"%str(word_lbk)+' '+"Right:%s"%str(word_rbk),
				(5, 100), font,0.7, (255, 0, 255),2)

		cv2.imshow("Result", IM)
		c = cv2.waitKey(5)
		if c>0:
			break
	time_end = time.time()
	time_cost = time_end - time_start
	plt.plot(pfn_lrbk)
	plt.show()
	print(time_cost)

### det: nx4 array of detections; det_info: nx6 array of detections; track_info mx7 list of track information; det_id: list of n length; track_id: list of m length; state_det: n array; state_track: m array
### det: xyxy; det_info: xywhvv; track_info: cv1w1h1vvf; num: count of total tracks
### c: cost; xy: coordinates; wh: width and height; vv: velocity vectors (horizontal and vertical); f: frames
### v1: delta v; w1: delta w; h1: delta h
### vp, wp, hp, fp for weights in calculating c; f_threshold, c_threshold for filter

import numpy as np
import cv2

### initialize track_id, track_info, num
def initialization():
    track_id=[]
    track_info=[] # should be an array
    num=0
    return track_id, track_info, num

### sample process
def process(det, track_id, track_info, num, f_threshold, c_threshold, vp, wp, hp, fp):    
    if not det:
        return track_id, track_info, num
    else:
        det_id, det_info, delta_det=get_np_matrix(det,track_info,vp,wp,hp,fp)
        state_det, state_track=assign(track_id, det_id, delta_det)
        track_id, track_info, num=update(track_id, track_info, num, state_track, state_det, det_info, delta_det, f_threshold, c_threshold)
        return track_id, track_info, num

### convert det to det_info and calculate delta_det (cost matrix)
def get_np_matrix(det,track_info,vp=2,wp=0.25,hp=0.25,fp=0):
    # initialization
    n=len(det)
    m=len(track_info)
    det_id=[i for i in range(0,n)]
    
    # det to det_info
    det_info=np.zeros((n,6))
    det_info[:,0]=(det[:,0]+det[:,2])/2
    det_info[:,1]=(det[:,1]+det[:,3])/2
    det_info[:,2]=det[:,2]-det[:,0]
    det_info[:,3]=det[:,3]-det[:,1]
    if m==0:
        return det_id, det_info, []
    
    # create cost matrix
    delta_det=np.zeros((m,n,6))
    track_info0=np.repeat(track_info,n,axis=0)
    track_info0=np.reshape(track_info0[:,:6],(m,n,6))
    det1=np.reshape(np.tile(det_info,(m,1)),(m,n,6))
    delta_det[:,:,4]=det1[:,:,0]-track_info0[:,:,0] # horizontal v
    delta_det[:,:,5]=det1[:,:,1]-track_info0[:,:,1] # vertical v
    delta_det[:,:,0]=(np.sqrt(det1[:,:,2]**2+det1[:,:,3]**2)+np.sqrt(track_info0[:,:,2]**2+track_info0[:,:,3]**2))/2
    delta_det[:,:,2]=abs(det1[:,:,2]-track_info0[:,:,2])/delta_det[:,:,0] # delta w
    delta_det[:,:,3]=abs(det1[:,:,3]-track_info0[:,:,3])/delta_det[:,:,0] # delta h
    delta_det[:,:,1]=np.linalg.norm(delta_det[:,:,4:]-track_info0[:,:,4:], axis=2)/delta_det[:,:,0] # delta v
    delta_det[:,:,0]=vp*delta_det[:,:,1]+wp*delta_det[:,:,2]+hp*delta_det[:,:,3] # c
    return det_id, det_info, delta_det

### assign detections to tracks
### state_track: row for assigned, -1 for unassigned; state_det: 0 for assigned, -1 for unassigned
def assign(track_id, det_id, delta_det):
    # initialization
    n=len(det_id)
    m=len(track_id)
    state_det=np.ones(n,dtype=int)*-1
    state_track=np.ones(m,dtype=int)*-1
    if n==0 or m==0:
        return state_det,state_track
    
    # assign initialization
    row_list=[[] for i in range(0,n)]
    new_track=[]
    delta_det_c=delta_det[:,:,0]
    indices=delta_det_c.argmin(1)
    col_unassigned=[] # sub track_id
    row_unassigned=[] # sub det_id
    col=0
    # get possible assignments for every detections
    for i in indices:
        row_list[i].append(col)
        col+=1
    row=0
    for i in row_list:
        # one detection per track
        if len(i)==1:
            state_track[i[0]]=det_id[row]
            state_det[row]=0
        # one detection multiple tracks
        elif len(i)>1:
            compare_list=delta_det[i,row,1]
            min_id=compare_list.argmin()
            state_track[i.pop(min_id)]=det_id[row]
            state_det[row]=0
            col_unassigned+=i
        # unassigned detections
        else:
            row_unassigned.append(row)
        row+=1
    # assign recursively if still assignable
    if row_unassigned and col_unassigned:
        delta_det_u=delta_det[col_unassigned][:,row_unassigned] # sub delta_det
        state_det[row_unassigned], state_track[col_unassigned]=assign(col_unassigned, row_unassigned, delta_det_u)
    return state_det, state_track
    
### update track_id and track_info using state_det and state_track
def update(track_id, track_info, num, state_track, state_det, det_info, delta_det, f_threshold=12, c_threshold=100):
    # update tracks with assigned detections
    col=0
    for i in state_track:
        if i==-1:
            track_info[col][-1]+=1
        elif delta_det[col,i,0]>c_threshold:
            track_info[col][-1]+=1
            state_det[i]=-1
        else:
            if track_info[col][-1]==0:
                track_info[col][:4]=det_info[i,:4].tolist()
                track_info[col][4:-1]=delta_det[col,i,4:].tolist()
            else:
                track_info[col][-1]=0
                track_info[col][:4]=det_info[i,:4].tolist()
                track_info[col][4:-1]=delta_det[col,i,4:].tolist()
        col+=1
    
    # create new tracks for unassigned detections
    new_track_id=[]
    new_track_info=[]
    row=0
    for i in state_det:
        if i==-1:
            num+=1
            new_track_id.append(num)
            new_info=det_info[row,:4].tolist()+[0,0,0]
            new_track_info.append(new_info)
        row+=1
    
    # delete lost tracks
    track_id+=new_track_id
    track_info+=new_track_info
    col=0
    for i in track_info:
        if i[-1]>f_threshold:
            track_id.pop(col)
            track_info.pop(col)
        col+=1
    return track_id, track_info, num

### label track_id on img
def label(img, track_id, track_info):
    size=22
    factor=0.1
    col=0
    for i in track_info:
        if i[-1]==0:
            x=i[0]-i[2]/2
            y=i[1]-i[3]/2            
            h=i[3]
            org=np.array([x,y+h*factor],dtype=int)
            scale=h*factor/size
            id=track_id[col]
            cv2.putText(img, '%d'%id, org, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale, color=(255,0,0), thickness=1, lineType=cv2.LINE_8)
        col+=1
    return img
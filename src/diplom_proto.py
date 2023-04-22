#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys, math

sys.path.append('../')
import gi

gi.require_version('Gst', '1.0')
import configparser
import json

import pyds
from common.bus_call    import bus_call
from common.is_aarch_64 import is_aarch64
from gi.repository      import GLib, Gst

import visualserver

import numpy as np
import cv2
import os
import stat


from common.is_aarch_64 import is_aarch64
from common.bus_call    import bus_call
from common.FPS         import PERF_DATA
from multiprocessing    import Process
from soundserver        import SoundServer
from camera_utils       import Camera, Vector3
from pathlib            import Path


import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# import scipy.misc
# rgb = scipy.misc.toimage(np_array)

PGIE_CLASS_ID_VEHICLE  = 0
PGIE_CLASS_ID_BICYCLE  = 1
PGIE_CLASS_ID_PERSON   = 2
PGIE_CLASS_ID_ROADSIGN = 3

pgie_classes_str = [
    "vechicle",
    "bichicle",
    "personichle",
    "roadsignichle"
]

serv = None #visualserver.Server([Camera(Vector3(8.0, 5.0, -3), 0.00, 0.00, 800, 600, 55)])

def osd_sink_pad_buffer_probe(pad,info,u_data):
    # print('pad:', pad)
    # for k in pad.__dict__:
    #     print(k, pad.__dict__[k])

    boxes = []
    ids   = []
    frame_number=0
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE:  0,
        PGIE_CLASS_ID_PERSON:   0,
        PGIE_CLASS_ID_BICYCLE:  0,
        PGIE_CLASS_ID_ROADSIGN: 0
    }
    num_rects=0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.glist_get_nvds_frame_meta()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            #frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        # print(
        #     'fnum:', frame_number, 
        #     'pad_index:', frame_meta.pad_index, 
        #     'source_id:', frame_meta.source_id
        # )
        num_rects = frame_meta.num_obj_meta
        
        l_obj=frame_meta.obj_meta_list

        # we need to extract faces from our image
        # code bnelow should save the image

        folder_name = '/tmp/face_from_frame_storage/'
        Path( folder_name ).mkdir( parents=True, exist_ok=True )
        os.chmod(folder_name, 0o0777 )


        # end of saving

        n_frame = None
        # print()
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                #obj_meta=pyds.glist_get_nvds_object_meta(l_obj.data)
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1

            # print(obj_meta.confidence)
            
            if obj_meta.class_id == PGIE_CLASS_ID_PERSON and obj_meta.confidence > 0.3:
                if True:
                    obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 0.5)
                    box = {
                        'top'   : obj_meta.rect_params.top,
                        'left'  : obj_meta.rect_params.left,
                        'width' : obj_meta.rect_params.width,
                        'height': obj_meta.rect_params.height
                    }
                    id = obj_meta.object_id
                    #print('id = ', id, box)
                    #вычисление координат объекта на экране:
                    b_y = box['top'] + box['height'] 
                    b_x = box['left']
                    boxes.append(visualserver.Vector3(b_x, b_y, 0))
                    ids.append(id)

                    if True: #frame_number > 16: #n_frame != None:
                        # print(f'=== {id} ===')
                        # id = obj_meta.object_id

                        if id not in serv.faced_ids or not serv.faced_ids[id]:
                            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                            frame_copy = np.array(n_frame, copy=True, order='C')
                            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
                                # last_track_id = track_id
                                # save_image = True
                            # print('Photo')
                            # img_path = "{}/frame_{}.{}.jpg".format(folder_name, frame_number, track_id)
                            Y = int(box['top'   ])
                            H = int(box['height'])
                            X = int(box['left'  ])
                            W = int(box['width' ])
                            frame_copy = frame_copy[Y:Y+H,X:X+W]
                            img_path = "{}/face_{}.jpg".format(folder_name, id)
                            cv2.imwrite(img_path, frame_copy)
                            print(f'Saved an img #{frame_number} of id {id}')
                            # serv.faced_ids[track_id] = None
            else:
                obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.5)
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}".format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE], obj_counter[PGIE_CLASS_ID_PERSON])

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        # print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
			
        #вычисляем координаты в комнате:
        #camr = cam.Camera(8.0, -3.0, 0.0, 2.0, 1.0, 2.0, 6.0, 1.0, 0.0, 0.0)
        #view = cam.Camera(1.0, 0.0, 0.0, 0.5, 0.1, 1.0, 4.0, 1.0, 0.0, 0.0)
        #ps = cam.place_objects(floor, boxes, camr, view)

        src_id = frame_meta.source_id
        serv.add_new_faces(src_id, ids, folder_name)
        serv.run(src_id, boxes, ids)

    #print(ps)

    # pj = []
    # for i, p in enumerate(ps):
    #     if p:
    #         d = p.dict().copy()
    #         d['id'] = ids[i]
    #         pj.append(d)

    # print(json.dumps(pj, indent=4))

    return Gst.PadProbeReturn.OK	

def checked_create(result, name = 'element'):
    if not result:
        sys.stderr.write(f"Unable to create {name}\n")
        exit(1)
    return result

def build_pipeline(sources):
    Gst.init(None)
    pipeline  = checked_create(Gst.Pipeline(), "pipeline")
    streammux = checked_create(Gst.ElementFactory.make("nvstreammux", "Stream-muxer"), "NvStreamMux")

    number_of_sources = len(sources)
    tiler=checked_create(Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler"), "tiler")
    
    tiler_rows=int(math.sqrt(number_of_sources))
    tiler_columns=int(math.ceil((1.0*number_of_sources)/tiler_rows))

    streammux.set_property('width', 1280 * tiler_rows)
    streammux.set_property('height', 960 * tiler_columns)
    # if src_type == 'cam':
    #     print("Playing cam %s " %src_path)
    #     # caps_v4l2src.set_property   ('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
    #     # caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    #     # source.set_property   ('device', src_path)
    #     streammux.set_property('width', 1280)
    #     streammux.set_property('height', 960)
    # else:
    #     print("Playing file %s " %src_path)
    #     source.set_property   ('location', src_path)
    #     streammux.set_property('width',  1920)
    #     streammux.set_property('height', 1080)

    streammux.set_property('batch-size', number_of_sources)
    streammux.set_property('batched-push-timeout', 4000000)
    mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
    # streammux.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_UNIFIED ))
    streammux.set_property("nvbuf-memory-type", mem_type)
    pipeline.add(streammux)

    cams = []
    for sidx, source_conf in enumerate(sources):
        src_type = source_conf[0].split('=')[0]
        src_path = source_conf[0].split('=')[1]
        cams.append(source_conf[1])

        # Create gstreamer elements
        # Create Pipeline element that will form a connection of other elements

        # Source element for reading from the file
        print("Creating Source \n ")
        
        if src_type == 'cam':
            source = Gst.ElementFactory.make("v4l2src", f"usb-cam-source-{sidx}")
            source.set_property('device', src_path)
            caps_v4l2src = checked_create(Gst.ElementFactory.make("capsfilter", f"v4l2src_caps{sidx}"), "v4l2src capsfilter")
            caps_v4l2src.set_property   ('caps', Gst.Caps.from_string(f"video/x-raw, framerate=20/1"))

            print("Creating Video Converter \n")
            vidconvsrc = checked_create(Gst.ElementFactory.make("videoconvert", f"convertor1_src{sidx}"), "videoconvert")

            # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
            nvvidconvsrc    = checked_create(Gst.ElementFactory.make("nvvideoconvert", f"convertor2_src{sidx}"), "Nvvideoconvert")
            caps_vidconvsrc = checked_create(Gst.ElementFactory.make("capsfilter", f"nvmm_caps{sidx}"), "capsfilter")
            caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
            pipeline.add(caps_v4l2src)
            pipeline.add(vidconvsrc)
            pipeline.add(nvvidconvsrc)
            pipeline.add(caps_vidconvsrc)
            # print('Using camera')
        else:
            source = Gst.ElementFactory.make("filesrc", "file-source")
            h264parser = checked_create(Gst.ElementFactory.make("h264parse", "h264-parser"), "h264 parser")
            decoder = checked_create(Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder"), "Nvv4l2 Decoder")
            pipeline.add(h264parser)
            pipeline.add(decoder)
            # print('Using file')
            

            print("Adding elements to Pipeline \n")
        pipeline.add(source)


        if src_type == 'cam':
            source.link      (caps_v4l2src)
            caps_v4l2src.link(vidconvsrc)
            vidconvsrc.link  (nvvidconvsrc)
            nvvidconvsrc.link(caps_vidconvsrc)
            pad_provider = caps_vidconvsrc
        else:
            source.link    (h264parser)
            h264parser.link(decoder)
            pad_provider = decoder

        sinkpad = checked_create(streammux.get_request_pad(f"sink_{sidx}"), "streammux")
        # srcpad = decoder.get_static_pad("src")
        srcpad = checked_create(pad_provider.get_static_pad("src"), "srcpad") 
        srcpad.link   (sinkpad)


    global serv
    serv = visualserver.Server(cams)
    

    # Use nvinfer to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    print('Creating pgie')
    pgie = checked_create(Gst.ElementFactory.make("nvinfer", "primary-inference"), "pgie")

    tracker = checked_create(Gst.ElementFactory.make("nvtracker", "tracker"), "tracker")

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = checked_create(Gst.ElementFactory.make("nvvideoconvert", "convertor"), "nvvidconv")
    # Create OSD to draw on the converted RGBA buffer
    nvosd = checked_create(Gst.ElementFactory.make("nvdsosd", "onscreendisplay"), "nvosd")

    # Finally render the osd output
    if is_aarch64():
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

    print("Creating EGLSink \n")
    sink = checked_create(Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer"), "egl sink")


    pgie.set_property     ('config-file-path',    "configs/pgie_config.ini")
    # Set sync = false to avoid late frame drops at the display-sink
    sink.set_property('sync', False)
    #Set properties of tracker
    config = configparser.ConfigParser()
    config.read    ('configs/tracker_config.ini')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process' :
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        if key == 'enable-past-frame' :
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame', tracker_enable_past_frame)

    # print("Adding elements to Pipeline \n")
    # pipeline.add(source)
    # if src_type == 'cam':
    #     pipeline.add(caps_v4l2src)
    #     pipeline.add(vidconvsrc)
    #     pipeline.add(nvvidconvsrc)
    #     pipeline.add(caps_vidconvsrc)
    # else:
    #     pipeline.add(h264parser)
    #     pipeline.add(decoder)
    
    # pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)
    if is_aarch64():
        pipeline.add(transform)

    # we link the elements together
    # v4l2src -> nvvideoconvert -> mux -> 
    # nvinfer -> nvvideoconvert -> nvosd -> video-renderer
    print("Linking elements in the Pipeline \n")
    # source.link(h264parser)
    # h264parser.link(decoder)

    streammux.link(pgie)
    pgie.link     (tracker)
    tracker.link  (nvvidconv)
    # nvvidconv.link(nvosd)


    tiler.set_property("rows",    tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", 1280*tiler_columns)
    tiler.set_property("height", 960*tiler_rows)
    if not is_aarch64():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        # mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)
    pipeline.add(tiler)
    # nvvidconv.link(tiler)
    # tiler.link(nvosd)
    nvvidconv.link(nvosd)

    if is_aarch64():
        nvosd.link    (transform)
        transform.link(tiler)
    else:
        nvosd.link(tiler)
    tiler.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop  ()
    bus = pipeline.get_bus()
    bus.add_signal_watch  ()
    bus.connect ("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = checked_create(nvosd.get_static_pad("sink"), "sink pad of nvosd")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


def main(args):
    # Check input arguments
    # sound = SoundServer()

    # sound_process = Process(target=sound.run)
    # sound_process.start()

    config = {}
    config_path = "camconf.json"
    if config_path:
        with open(config_path, 'r') as config_file:
            config = json.loads(config_file.read())

    
    sources = []
    for source_name in config:
        if source_name[0] not in ['-', '#']:
            source_config = config[source_name]
            print(source_name, source_config)
            source = 'cam=' + source_config['source']
            # print(source)
            cam = Camera(Vector3(*source_config['position']), *source_config['params'])
            sources.append([source, cam])
        # print(cam)

    for source in sources:
        print(source)
    build_pipeline(sources)

    # sound_process.terminate()
    return 0
    

if __name__ == '__main__':
    sys.exit(main(sys.argv))


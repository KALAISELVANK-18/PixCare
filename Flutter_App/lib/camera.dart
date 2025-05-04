import 'dart:ui';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:percent_indicator/circular_percent_indicator.dart';

import 'ShowImage.dart';

final user = FirebaseAuth.instance.currentUser;

// A widget that displays the picture taken by the user.
class DisplayPictureScreen extends StatelessWidget {
  final String imagePath;
  int post = 0;
  int indicator = 0;
  final Map<String, dynamic>? result;

  DisplayPictureScreen(
      {super.key, required this.imagePath, required this.result});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Center(child: const Text('Save report'))),
      // The image is stored as a file on the device. Use the Image.file
      // constructor with the given path to display the image.
      body: SingleChildScrollView(
        child: Column(
          children: [
            SizedBox(
              height: 10,
            ),
            Container(
              width: MediaQuery.of(context).size.width,
              child: Column(
                children: [
                  Container(
                    height: 300,
                    color: Color.fromARGB(255, 253, 229, 219),
                    child: Padding(
                      padding: const EdgeInsets.all(15.0),
                      child: Container(
                          height: 100,
                          width: 500,
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(10),
                            image: DecorationImage(
                              fit: BoxFit.cover,
                              image: FileImage(
                                File(imagePath),
                              ),
                            ),
                          ),
                          child: SizedBox()),
                    ),
                  ),
                  SizedBox(
                    height: 10,
                  ),
                  Container(
                    child: Column(
                      children: [
                        Padding(
                          padding: const EdgeInsets.only(top: 13),
                          child: Text(
                            "Explore the Top-3 results and choose most similar",
                            style: TextStyle(color: Colors.grey, fontSize: 13),
                          ),
                        ),
                        Text(
                          "to your symptoms to get personal advice.",
                          style: TextStyle(color: Colors.grey, fontSize: 13),
                        ),
                        (result!['result'].length > 1)
                            ? ListView.builder(
                                physics: NeverScrollableScrollPhysics(),
                                shrinkWrap: true,
                                itemCount: result!["result"].length,
                                itemBuilder: (context, ind) {
                                  return (ind == 0)
                                      ? Padding(
                                          padding: const EdgeInsets.only(
                                              left: 10, right: 10, top: 20),
                                          child: Container(
                                            // decoration: BoxDecoration(
                                            //   borderRadius: BorderRadius.all(Radius.circular(30)),
                                            // ),

                                            width: MediaQuery.of(context)
                                                    .size
                                                    .width *
                                                0.95,
                                            decoration: BoxDecoration(
                                              borderRadius: BorderRadius.all(
                                                  Radius.circular(15)),
                                              color: Color.fromARGB(
                                                  255, 253, 229, 219),
                                            ),
                                            child: Row(
                                              children: [
                                                Padding(
                                                  padding:
                                                      const EdgeInsets.all(8.0),
                                                  child:
                                                      new CircularPercentIndicator(
                                                    radius: 35.0,
                                                    lineWidth: 5.0,
                                                    percent: double.parse(
                                                        result!["result"][ind]
                                                            [1]),
                                                    center: new Text(
                                                        "${double.parse(result!["result"][ind][1])}%"),
                                                    progressColor: Colors.green,
                                                  ),
                                                ),
                                                Padding(
                                                  padding:
                                                      const EdgeInsets.only(
                                                          left: 5),
                                                  child: Column(
                                                    mainAxisAlignment:
                                                        MainAxisAlignment.start,
                                                    crossAxisAlignment:
                                                        CrossAxisAlignment
                                                            .start,
                                                    children: [
                                                      Container(
                                                          width: 200,
                                                          child: Text(
                                                            result!["result"]
                                                                [ind][0],
                                                            style: TextStyle(
                                                                color: Colors
                                                                    .deepPurpleAccent,
                                                                fontSize: 16),
                                                          )),
                                                      SizedBox(
                                                        height: 4,
                                                      ),
                                                      Row(
                                                        mainAxisAlignment:
                                                            MainAxisAlignment
                                                                .start,
                                                        children: [
                                                          //Text("No Pathologies",style: TextStyle(color: Colors.grey,fontSize: 12),),
                                                        ],
                                                      ),
                                                      SizedBox(
                                                        height: 8,
                                                      ),
                                                      Text(
                                                        "Description",
                                                        style: TextStyle(
                                                            color: Colors
                                                                .deepOrangeAccent),
                                                      )
                                                    ],
                                                  ),
                                                ),
                                                SizedBox(
                                                  width: 50,
                                                ),
                                                Icon(
                                                  Icons.chevron_right,
                                                  size: 30,
                                                  color:
                                                      Colors.deepOrangeAccent,
                                                )
                                              ],
                                            ),
                                          ),
                                        )
                                      : Row(
                                          children: [
                                            Padding(
                                              padding: const EdgeInsets.only(
                                                  left: 18, top: 30),
                                              child:
                                                  new CircularPercentIndicator(
                                                radius: 25.0,
                                                lineWidth: 5.0,
                                                percent: double.parse(
                                                    result!["result"][ind][1]),
                                                center: new Text(
                                                    "${double.parse(result!["result"][ind][1])}%"),
                                                progressColor: Colors.green,
                                              ),
                                            ),
                                            Padding(
                                              padding: const EdgeInsets.only(
                                                  left: 15),
                                              child: Column(
                                                children: [
                                                  Container(
                                                      width: 200,
                                                      child: Text(
                                                        result!["result"][ind]
                                                            [0],
                                                        style: TextStyle(
                                                            color: Colors.grey),
                                                      )),
                                                  SizedBox(
                                                    height: 5,
                                                  ),
                                                  Text(
                                                    "Benign Lesions",
                                                    style: TextStyle(
                                                        color: Colors.grey,
                                                        fontSize: 12),
                                                  )
                                                ],
                                                crossAxisAlignment:
                                                    CrossAxisAlignment.start,
                                              ),
                                            ),
                                            SizedBox(
                                              width: 150,
                                            ),
                                            Icon(
                                              Icons.chevron_right,
                                              color: Colors.grey,
                                              size: 25,
                                            )
                                          ],
                                        );
                                })
                            : Padding(
                                padding:
                                    const EdgeInsets.only(left: 5, top: 20),
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.start,
                                  crossAxisAlignment: CrossAxisAlignment.center,
                                  children: [
                                    Container(
                                        width: 200,
                                        child: Text(
                                          result!["result"][0],
                                          style: TextStyle(
                                              color: Colors.deepPurpleAccent,
                                              fontSize: 20),
                                        )),
                                    SizedBox(
                                      height: 4,
                                    ),
                                    Row(
                                      mainAxisAlignment:
                                          MainAxisAlignment.start,
                                      children: [
                                        //Text("No Pathologies",style: TextStyle(color: Colors.grey,fontSize: 12),),
                                      ],
                                    ),
                                    SizedBox(
                                      height: 8,
                                    ),
                                    //Text("Description",style: TextStyle(color: Colors.deepOrangeAccent),)
                                  ],
                                ),
                              ),
                      ],
                    ),
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.only(
                        topLeft: Radius.circular(40),
                        topRight: Radius.circular(40),
                      ),
                    ),
                  ),
                  SizedBox(
                    height: 20,
                  ),
                  Text(
                    "New check",
                    style: TextStyle(
                        color: Colors.deepOrangeAccent,
                        fontSize: 18,
                        fontWeight: FontWeight.bold),
                  ),
                  SizedBox(
                    height: 30,
                  ),
                  Text(
                    "Skivine does not diagnose. If in doubt,",
                    style: TextStyle(color: Colors.grey, fontSize: 13),
                  ),
                  Text(
                    "seek the advice of a specialist doctor.",
                    style: TextStyle(color: Colors.grey, fontSize: 13),
                  ),
                  SizedBox(
                    height: 20,
                  ),
                  Text(
                    "User Agreement",
                    style: TextStyle(color: Colors.deepOrangeAccent),
                  )
                ],
              ),
            ),
            Container(
              width: 140,
              child: ElevatedButton(
                  style: ButtonStyle(
                      backgroundColor:
                          MaterialStateProperty.all(Colors.redAccent)),
                  onPressed: () async {
                    if (post == 0) {
                      post = 1;
                      try {
                        final c = DateTime.timestamp();

                        final refe = FirebaseStorage.instance
                            .ref()
                            .child(FirebaseAuth.instance.currentUser!.uid)
                            .child(c.toString())
                            .child("tum");

                        await refe.putFile(File(imagePath),
                            SettableMetadata(contentType: "image/png"));

                        final downloadUrl1 = await FirebaseStorage.instance
                            .ref()
                            .child(FirebaseAuth.instance.currentUser!.uid)
                            .child(c.toString())
                            .child("tum")
                            .getDownloadURL();
                        final String url1 = downloadUrl1.toString();

                        await FirebaseFirestore.instance
                            .collection(FirebaseAuth.instance.currentUser!.email
                                .toString())
                            .doc("posts")
                            .collection("posts")
                            .add({
                          "time": DateTime.timestamp(),
                          "uid": FirebaseAuth.instance.currentUser!.uid,
                          "title": result!["result"][0],
                          "picture_url": url1,
                        });
                        post = 2;
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            backgroundColor: Colors.greenAccent,
                            content: Text('Saved!'),
                          ),
                        );

                        Navigator.pop(context);
                        Navigator.pop(context);
                      } catch (error) {
                        post = 0;
                        print(post);
                        print(error);
                      }
                    }
                  },
                  child: Center(
                    child: Text("save",
                        textAlign: TextAlign.center,
                        style: GoogleFonts.stylish(
                          textStyle: TextStyle(
                              fontSize: 27,
                              fontWeight: FontWeight.bold,
                              color: Colors.white),
                        )),
                  )),
            )
          ],
        ),
      ),
    );
  }
}

class TakePictureScreen extends StatefulWidget {
  List<CameraDescription>? cameras;
  final String? user;
  final String skin;
  Map<String, String> questions ;
  TakePictureScreen({super.key, required this.cameras,required this.user,required this.skin,required this.questions});

  @override
  TakePictureScreenState createState() => TakePictureScreenState();
}

class TakePictureScreenState extends State<TakePictureScreen>
    with TickerProviderStateMixin {
  late CameraController _controller;
  late double minzoomoffset = 1;
  int indicator = 0;
  late double maxzoomoffset = 10;
  late FutureOr<void> _initializeControllerFuture;
  late TabController tabController;
  TextEditingController email = new TextEditingController();
  TextEditingController pass = new TextEditingController();
  TextEditingController conpass = new TextEditingController();
  int currentTabIndex = 0;
  double _currentZoomOffset = 1;
  @override
  void initState() {
    super.initState();
    _controller = CameraController(widget.cameras![0], ResolutionPreset.high,
        enableAudio: false);
    // minzoomoffset=_controller.getMaxZoomLevel();
    // minzoomoffset=_controller.getMinZoomLevel();
    _controller.initialize().then((_) {
      // Set autofocus mode
      _controller.setFlashMode(FlashMode.off);
      if (!mounted) {
        return;
      }

      setState(() {});
    });
  }

  @override
  void dispose() {
    // Dispose of the controller when the widget is disposed.
    _controller.dispose();
    super.dispose();
  }

  void onTabChange() {
    setState(() {
      currentTabIndex = tabController.index;
    });
  }

  Map<String, dynamic> responsee = {};
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SingleChildScrollView(
        child: Column(
          children: [
            SizedBox(
              height: 30,
            ),
            Padding(
              padding: const EdgeInsets.all(10),
              child: Container(
                decoration:
                    BoxDecoration(borderRadius: BorderRadius.circular(20)),
                child: (_controller.value.isInitialized)
                    ? Container(
                        child: Center(
                        child: Container(
                          child: ClipRRect(
                              borderRadius: BorderRadius.circular(16.0),
                              child: CameraPreview(_controller)),
                        ),
                      ))
                    : CircularProgressIndicator(),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text('-', style: TextStyle(fontSize: 40)),
                  Expanded(
                    child: Slider(
                      value: _currentZoomOffset,
                      min: minzoomoffset,
                      max: maxzoomoffset,
                      onChanged: (newv) {
                        setState(() {
                          _currentZoomOffset = newv;
                          print(_currentZoomOffset);
                          if (newv > minzoomoffset && newv < maxzoomoffset)
                            _controller.setZoomLevel(newv);
                        });
                      },
                    ),
                  ),
                  Text(
                    '+',
                    style: TextStyle(fontSize: 30),
                  ),
                ],
              ),
            )
          ],
        ),
      ),
      floatingActionButton: Padding(
        padding: const EdgeInsets.only(bottom: 30),
        child: FloatingActionButton.extended(
          label: Text("Click!"),

          // Provide an onPressed callback.
          onPressed: () async {
            //!email.text.isEmpty && !pass.text.isEmpty
            if (email.text.isEmpty && pass.text.isEmpty && indicator == 0) {
              try {
                // Attempt to take a picture and get the file image
                // where it was saved.
                setState(() {
                  indicator = 1;
                });
                final image = await _controller.takePicture();
                Navigator.of(context).push(MaterialPageRoute(
                    builder: (context) => ShowImage(imagePath: image.path,userType: widget.user,skin : widget.skin,questions: widget.questions,)));
                setState(() {
                  indicator = 0;
                });
                // showDialog<String>(
                //   context: context,
                //   builder: (BuildContext context) => AlertDialog(
                //     title: Text('there is ${dataMap?["result"][0]}'),
                //
                //     actions: <Widget>[
                //       TextButton(
                //         onPressed: () => Navigator.pop(context, 'Cancel'),
                //         child: const Text('Cancel'),
                //       ),
                //       TextButton(
                //         onPressed: () => Navigator.pop(context, 'OK'),
                //         child: const Text('OK'),
                //       ),
                //     ],
                //   ),
                // );

                if (!mounted) return;
              } catch (e) {
                // If an error occurs, log the error to the console.
                print(e);
                ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                  content: const Text('Issue in connecting cameras'),
                  action: SnackBarAction(
                    label: 'Undo',
                    onPressed: () {
                      // Some code to undo the change.
                    },
                  ),
                ));
              }
            }
          },
          icon: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.camera_alt),
              (indicator == 1)
                  ? Container(
                      height: 10, width: 10, child: CircularProgressIndicator())
                  : SizedBox()
            ],
          ),
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
    );
  }

  FutureOr<void> _showMyDialog(String targ, String val) async {
    return showDialog<void>(
      context: context,
      barrierDismissible: false, // user must tap button!
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('AlertDialog Title'),
          content: const SingleChildScrollView(
            child: ListBody(
              children: <Widget>[],
            ),
          ),
          actions: <Widget>[
            TextButton(
              child: const Text('Ok'),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }
}

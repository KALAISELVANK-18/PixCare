
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:camera/camera.dart';
import 'package:carousel_slider/carousel_slider.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:dots_indicator/dots_indicator.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:intl/intl.dart';
import 'package:novalabs/BodyModel.dart';
import 'package:novalabs/Details2.dart';
import 'package:novalabs/Dummy.dart';
import 'package:novalabs/IotDevice.dart';
import 'package:novalabs/Notifications.dart';
import 'package:novalabs/Questions.dart';

import 'package:novalabs/Signin.dart';
import 'package:persistent_bottom_nav_bar/persistent_tab_view.dart';
import 'package:pie_chart/pie_chart.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'Report.dart';
import 'firebase_options.dart';
import 'package:firebase_messaging/firebase_messaging.dart';


List<CameraDescription>? cameras;
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  // If you're going to use other Firebase services in the background, such as Firestore,
  // make sure you call `initializeApp` before using other Firebase services.
  await Firebase.initializeApp();

  print("Handling a background message: ${message.messageId}");
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  await FirebaseMessaging.instance.getInitialMessage();

  FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);
  cameras = await availableCameras();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        debugShowCheckedModeBanner: false,
        title: 'Novalabs',
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
          useMaterial3: true,
        ),

        home:
      (FirebaseAuth.instance.currentUser == null)
            ? Signin() : HomePage(),
    );
  }
  Future<bool> checkIfUserIsSignedIn() async {
    // Replace this with your logic to check if the user is signed in
    // For example, you can use FirebaseAuth to check if the current user is not null
    // This method should return a boolean indicating whether the user is signed in or not
    // Example:
    await for (var user in FirebaseAuth.instance.authStateChanges()) {
      if (user != null) {
        return true;
      }
    }
    return false;
  }
}

// class MyHomePage extends StatefulWidget {
//   const MyHomePage({super.key});

//   @override
//   State<MyHomePage> createState() => _MyHomePageState();
// }

// class _MyHomePageState extends State<MyHomePage> {
//   int _counter = 0;

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       backgroundColor: Colors.white,
//       appBar: AppBar(
//         actions: [
//           Padding(
//             padding: const EdgeInsets.all(8.0),
//             child: IconButton(
//                 onPressed: () {
//                   PersistentNavBarNavigator.pushNewScreen(
//                     context,
//                     screen: Profile(),
//                     withNavBar: true,
//                     pageTransitionAnimation: PageTransitionAnimation.fade,
//                   );
//                 },
//                 icon: Icon(
//                   Icons.person_pin_circle_rounded,
//                   size: 33,
//                   color: Color.fromARGB(255, 241, 194, 125),
//                 )),
//           )
//         ],
//         elevation: 0,

//         // TRY THIS: Try changing the color here to a specific color (to
//         // Colors.amber, perhaps?) and trigger a hot reload to see the AppBar
//         // change color while the other colors stay the same.
//         backgroundColor: Colors.white,
//         // Here we take the value from the MyHomePage object that was created by
//         // the App.build method, and use it to set our appbar title.
//         title: Row(
//           crossAxisAlignment: CrossAxisAlignment.center,
//           children: [
//             Text(
//               "N",
//               style: GoogleFonts.aBeeZee(
//                   textStyle:
//                       TextStyle(fontWeight: FontWeight.bold, fontSize: 30)),
//             ),
//             Image.asset(
//               "images/nova.png",
//               height: 30,
//               width: 30,
//             ),
//             Text("va",
//                 style: GoogleFonts.kanit(
//                   textStyle: TextStyle(fontSize: 25),
//                 )),
//             Padding(
//               padding: const EdgeInsets.only(top: 8.0),
//               child: Text(
//                 "l",
//                 style: GoogleFonts.sacramento(
//                   textStyle: TextStyle(
//                       fontSize: 27,
//                       fontWeight: FontWeight.bold,
//                       color: Colors.amber),
//                 ),
//               ),
//             ),
//             Padding(
//               padding: const EdgeInsets.only(top: 8.0),
//               child: Text(
//                 "a",
//                 style: GoogleFonts.sacramento(
//                   textStyle: TextStyle(
//                       fontSize: 27,
//                       fontWeight: FontWeight.bold,
//                       color: Colors.redAccent),
//                 ),
//               ),
//             ),
//             Padding(
//               padding: const EdgeInsets.only(top: 8.0),
//               child: Text(
//                 "bs",
//                 style: GoogleFonts.sacramento(
//                   textStyle: TextStyle(
//                       fontSize: 27,
//                       fontWeight: FontWeight.bold,
//                       color: Colors.blue),
//                 ),
//               ),
//             ),
//           ],
//         ),
//       ),
//       body: SingleChildScrollView(
//         child: Center(
//           // Center is a layout widget. It takes a single child and positions it
//           // in the middle of the parent.
//           child: Padding(
//             padding: const EdgeInsets.all(8.0),
//             child: Column(
//               mainAxisAlignment: MainAxisAlignment.start,
//               children: <Widget>[
//                 Stack(children: [
//                   Container(
//                     decoration: BoxDecoration(
//                       gradient: LinearGradient(colors: [
//                         Color.fromARGB(150, 255, 219, 172),
//                         Color.fromARGB(255, 255, 219, 172),
//                       ]),
//                       borderRadius: BorderRadius.circular(10),
//                     ),
//                     child: Column(
//                       children: [
//                         Row(
//                           mainAxisAlignment: MainAxisAlignment.spaceBetween,
//                           children: [
//                             Column(
//                               mainAxisAlignment: MainAxisAlignment.center,
//                               crossAxisAlignment: CrossAxisAlignment.center,
//                               children: [
//                                 Padding(
//                                   padding: const EdgeInsets.all(8.0),
//                                   child: Image.asset(
//                                     "images/disease.png",
//                                     width: 100,
//                                     height: 100,
//                                     scale: 2,
//                                   ),
//                                 ),
//                                 Padding(
//                                   padding: const EdgeInsets.all(0),
//                                   child: Center(
//                                     child: Container(
//                                         width: 110,
//                                         child: Text(
//                                           "you need personalized skin check up?",
//                                           style: GoogleFonts.aBeeZee(
//                                               textStyle: TextStyle(
//                                                   fontWeight: FontWeight.bold)),
//                                           textAlign: TextAlign.center,
//                                         )),
//                                   ),
//                                 )
//                               ],
//                             ),
//                             Column(
//                               mainAxisAlignment: MainAxisAlignment.center,
//                               crossAxisAlignment: CrossAxisAlignment.center,
//                               children: [
//                                 Padding(
//                                   padding: const EdgeInsets.only(left: 0),
//                                   child: Image.asset(
//                                     "images/diseases2.png",
//                                     width: 120,
//                                     height: 120,
//                                     scale: 1,
//                                   ),
//                                 ),
//                                 Padding(
//                                   padding: const EdgeInsets.all(0),
//                                   child: Container(
//                                       width: 120,
//                                       child: Text(
//                                         "Are you travelling daily to clinic?",
//                                         style: GoogleFonts.aBeeZee(
//                                             textStyle: TextStyle(
//                                           fontWeight: FontWeight.bold,
//                                         )),
//                                         textAlign: TextAlign.center,
//                                       )),
//                                 )
//                               ],
//                             ),
//                             Column(
//                               mainAxisAlignment: MainAxisAlignment.center,
//                               crossAxisAlignment: CrossAxisAlignment.center,
//                               children: [
//                                 Padding(
//                                   padding: const EdgeInsets.only(right: 8.0),
//                                   child: Image.asset(
//                                     "images/diseases3.png",
//                                     width: 120,
//                                     height: 120,
//                                     scale: 1,
//                                   ),
//                                 ),
//                                 Padding(
//                                   padding: const EdgeInsets.only(left: 0),
//                                   child: Container(
//                                       width: 110,
//                                       child: Text(
//                                         "Are you confused? Dont worry!",
//                                         style: GoogleFonts.aBeeZee(
//                                             textStyle: TextStyle(
//                                                 fontWeight: FontWeight.bold)),
//                                         textAlign: TextAlign.center,
//                                       )),
//                                 )
//                               ],
//                             ),
//                           ],
//                         ),
//                         SizedBox(
//                           height: 50,
//                         )
//                       ],
//                     ),
//                   ),
//                   Center(
//                     child: Padding(
//                       padding: const EdgeInsets.only(top: 190),
//                       child: Container(
//                         width: 340,
//                         decoration: BoxDecoration(
//                           boxShadow: [
//                             BoxShadow(
//                               color: Colors.deepOrangeAccent,
//                               blurRadius: 10.0, // soften the shadow
//                               spreadRadius: 3.0, //extend the shadow
//                               offset: Offset(
//                                 5.0, // Move to right 5  horizontally
//                                 5.0, // Move to bottom 5 Vertically
//                               ),
//                             )
//                           ],
//                           gradient: LinearGradient(colors: [
//                             Color.fromARGB(255, 238, 75, 43),
//                             Colors.redAccent.shade200
//                           ]),
//                           borderRadius: BorderRadius.circular(10),
//                         ),
//                         child: Column(
//                           children: [
//                             Row(
//                               mainAxisAlignment: MainAxisAlignment.spaceBetween,
//                               children: [
//                                 Padding(
//                                   padding: const EdgeInsets.all(8.0),
//                                   child: Row(
//                                     mainAxisAlignment: MainAxisAlignment.center,
//                                     crossAxisAlignment:
//                                         CrossAxisAlignment.center,
//                                     children: [
//                                       Padding(
//                                         padding:
//                                             const EdgeInsets.only(right: 8.0),
//                                         child: Image.asset(
//                                           "images/solution.png",
//                                           width: 120,
//                                           height: 120,
//                                           scale: 1,
//                                         ),
//                                       ),
//                                       Padding(
//                                         padding: const EdgeInsets.only(left: 6),
//                                         child: Container(
//                                             width: 190,
//                                             child: Text(
//                                               "Dont worry! Click the booster appears down and take pic of skin spot to get personalized report!",
//                                               style: GoogleFonts.aBeeZee(
//                                                   textStyle: TextStyle(
//                                                       color: Colors.white,
//                                                       fontWeight:
//                                                           FontWeight.bold)),
//                                               textAlign: TextAlign.center,
//                                             )),
//                                       )
//                                     ],
//                                   ),
//                                 ),
//                               ],
//                             ),
//                             Padding(
//                               padding: const EdgeInsets.only(left: 0),
//                               child: Padding(
//                                 padding:
//                                     const EdgeInsets.symmetric(horizontal: 8.0),
//                                 child: Row(
//                                   mainAxisAlignment: MainAxisAlignment.center,
//                                   children: [
//                                     Text(
//                                       "Novalabs ai powered ",
//                                       style: GoogleFonts.alef(
//                                           textStyle: TextStyle(
//                                         color: Colors.white,
//                                       )),
//                                       textAlign: TextAlign.center,
//                                     ),
//                                     Icon(
//                                       Icons.rocket_launch_rounded,
//                                       color: Color.fromARGB(255, 255, 219, 172),
//                                     )
//                                   ],
//                                 ),
//                               ),
//                             ),
//                             SizedBox(
//                               height: 10,
//                             ),
//                           ],
//                         ),
//                       ),
//                     ),
//                   )
//                 ]),
//                 Padding(
//                   padding: const EdgeInsets.only(top: 57.0),
//                   child: Column(
//                     children: [
//                       Container(
//                         child: ElevatedButton(
//                             style: ButtonStyle(
//                               shape: MaterialStateProperty.all(CircleBorder()),
//                               padding:
//                                   MaterialStateProperty.all(EdgeInsets.all(20)),
//                               backgroundColor: MaterialStateProperty.all(
//                                   Colors.white), // <-- Button color
//                               overlayColor:
//                                   MaterialStateProperty.resolveWith<Color?>(
//                                       (states) {
//                                 if (states.contains(MaterialState.pressed))
//                                   return Colors.white;
//                               }),
//                             ),
//                             onPressed: () {
//                               Navigator.push(
//                                   context,
//                                   MaterialPageRoute(
//                                       builder: (context) => TakePictureScreen(
//                                             cameras: cameras,
//                                           )));
//                             },
//                             child: Image.asset(
//                               "images/solu.png",
//                               width: 50,
//                               height: 50,
//                             )),
//                       ),
//                       Padding(
//                         padding: const EdgeInsets.all(8.0),
//                         child: Text(
//                           "Boost🔥",
//                           style: GoogleFonts.stylish(
//                             textStyle: TextStyle(
//                                 fontSize: 22,
//                                 fontWeight: FontWeight.bold,
//                                 color: Colors.redAccent),
//                           ),
//                         ),
//                       )
//                     ],
//                   ),
//                 ),
//                 SizedBox(),
//                 Row(
//                   mainAxisAlignment: MainAxisAlignment.start,
//                   children: [
//                     Padding(
//                       padding: const EdgeInsets.all(4.0),
//                       child: Text("Highlights",
//                           style: GoogleFonts.stylish(
//                             textStyle: TextStyle(
//                                 fontSize: 30,
//                                 fontWeight: FontWeight.bold,
//                                 color: Colors.redAccent),
//                           )),
//                     ),
//                   ],
//                 ),
//                 Container(
//                   decoration: BoxDecoration(
//                     gradient: LinearGradient(colors: [
//                       Color.fromARGB(220, 255, 219, 172),
//                       Color.fromARGB(220, 255, 219, 172),
//                     ]),
//                     borderRadius: BorderRadius.circular(10),
//                   ),
//                   child: Column(
//                     children: [
//                       Row(
//                         mainAxisAlignment: MainAxisAlignment.center,
//                         crossAxisAlignment: CrossAxisAlignment.center,
//                         children: [
//                           Padding(
//                             padding: const EdgeInsets.all(8.0),
//                             child: Image.asset(
//                               "images/location.png",
//                               width: 155,
//                               height: 155,
//                               scale: 2,
//                             ),
//                           ),
//                           Padding(
//                             padding: const EdgeInsets.symmetric(vertical: 20),
//                             child: Center(
//                               child: Container(
//                                   width: 200,
//                                   child: Text(
//                                     "Find novalabs nearest dermatologists and get speacialized Treatment with our product especially for cancers!",
//                                     style: GoogleFonts.aBeeZee(
//                                         textStyle: TextStyle(
//                                             fontWeight: FontWeight.bold)),
//                                     textAlign: TextAlign.center,
//                                   )),
//                             ),
//                           ),
//                         ],
//                       )
//                     ],
//                   ),
//                 ),
//                 SizedBox(
//                   height: 30,
//                 ),
//                 Container(
//                   decoration: BoxDecoration(
//                     gradient: LinearGradient(colors: [
//                       Color.fromARGB(150, 198, 134, 66),
//                       Color.fromARGB(150, 198, 134, 66),
//                     ]),
//                     borderRadius: BorderRadius.circular(10),
//                   ),
//                   child: Column(
//                     children: [
//                       Row(
//                         mainAxisAlignment: MainAxisAlignment.center,
//                         crossAxisAlignment: CrossAxisAlignment.center,
//                         children: [
//                           Padding(
//                             padding: const EdgeInsets.all(8.0),
//                             child: Image.asset(
//                               "images/iot.png",
//                               width: 155,
//                               height: 155,
//                               scale: 2,
//                             ),
//                           ),
//                           Padding(
//                             padding: const EdgeInsets.all(0),
//                             child: Center(
//                               child: Container(
//                                   width: 200,
//                                   child: Text(
//                                     "Buy nova product to get specialized skin care!",
//                                     style: GoogleFonts.aBeeZee(
//                                         textStyle: TextStyle(
//                                             fontWeight: FontWeight.bold)),
//                                     textAlign: TextAlign.center,
//                                   )),
//                             ),
//                           ),
//                         ],
//                       )
//                     ],
//                   ),
//                 ),
//               ],
//             ),
//           ),
//         ),
//       ),
//       // floatingActionButton: FloatingActionButton(
//       //   onPressed:(){
//       //
//       //   },
//       //   tooltip: 'Increment',
//       //   child: const Icon(Icons.add),
//       // ), // This trailing comma makes auto-formatting nicer for build methods.
//     );
//   }
// }
List<String> listval1 = <String>[
  'Head(सिर)',
  'Body(शरीर)',
  'Left Hand(बाएं हाथ)',
  'Right Hand(दाएं हाथ)',
  'Left Leg(बाएं पैर)',
  'Right Leg(दाएं पैर)'
];
List<String> listval1Hindi = <String>[
  'सिर',
  'शरीर',
  'बाएं हाथ',
  'दाएं हाथ',
  'बाएं पैर',
  'दाएं पैर',
];


class HomePage extends StatefulWidget {
  const HomePage({Key? key}) : super(key: key);

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _currentIndex = 0;
  int _currentIndex1 = 0;

  final List<String> images = [
    "list.png",
    "report.png",
    "3dmap.jpg",
  ];

  final List<String> sliderItems = [
    "list.png",
    "report.png",
    "3dmap.jpg",
  ];

  final dataMap = <String, double>{
    "Disease Found": 10,
    "No Disease Found": 10,
  };

  final colorList = <Color>[Colors.greenAccent, Colors.redAccent];
  String formatTimestamp(int timestamp) {
    var format = new DateFormat('dd-MMM-yyyy');
    var date = new DateTime.fromMillisecondsSinceEpoch(timestamp * 1000);
    return format.format(date);
  }

  @override
  String? mtoken = "";
  void initState() {
    super.initState();
    req();
    get();
    initInfo();
    getUserrType();
    getLanguage();
  }
  late File _image;
  final picker = ImagePicker();
  late String? userType;
  initInfo() async {
    FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
        FlutterLocalNotificationsPlugin();
    var androidInitialize =
        const AndroidInitializationSettings('@mipmap/ic_launcher');

    var initializationSettings = InitializationSettings(
      android: androidInitialize,
    );
    void _onNotificationTap(NotificationResponse notificationResponse) {
      // Handle notification tap

      // Navigator.push(
      //   context,
      //   MaterialPageRoute(builder: (context) => const Universal()),
      // );
    }
    flutterLocalNotificationsPlugin.initialize(
      initializationSettings,
      onDidReceiveNotificationResponse: _onNotificationTap,
    );

    FirebaseMessaging.onMessage.listen((RemoteMessage message) async {
      print('Got a message whilst in the foreground!');
      print('Message data: ${message.data}');

      if (message.notification != null) {
        print(
            'Message also contained a notification: ${message.notification!.title}');
      }
      BigTextStyleInformation bigTextStyleInformation = BigTextStyleInformation(
        message.notification!.body.toString(),
        htmlFormatBigText: true,
        contentTitle: message.notification!.title.toString(),
        htmlFormatContentTitle: true,
      );
      AndroidNotificationDetails androidNotificationDetails =
          AndroidNotificationDetails(
        'dbnoti',
        'dbnoti',
        importance: Importance.max,
        styleInformation: bigTextStyleInformation,
        priority: Priority.high,
        playSound: true,
      );
      NotificationDetails notificationDetails =
          NotificationDetails(android: androidNotificationDetails);
      await flutterLocalNotificationsPlugin.show(0, message.notification?.title,
          message.notification?.body, notificationDetails,
          payload: message.data['title']);
    });
  }

  void get() async {
    final fcmToken = await FirebaseMessaging.instance.getToken();
    mtoken = fcmToken!;

    String user = await FirebaseAuth.instance.currentUser!.email.toString();
    await FirebaseFirestore.instance.collection("Usertokens").doc(user).set(
        {"tokens": mtoken, "user": user}).then((value) => print("Successful"));
  }
  String? language;

  Future<String?> getLanguage() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      language = prefs.getString("userLanguage");
    });
    return null;
  }
  void req() async {
    FirebaseMessaging messaging = FirebaseMessaging.instance;
    NotificationSettings settings = await messaging.requestPermission(
      alert: true,
      announcement: false,
      badge: true,
      carPlay: false,
      criticalAlert: false,
      provisional: false,
      sound: true,
    );
    if (settings.authorizationStatus == AuthorizationStatus.authorized) {
      print('User granted');
    } else if (settings.authorizationStatus ==
        AuthorizationStatus.provisional) {
      print('User granted provisional');
    } else {
      print('User declined');
    }
  }

  Future<String?> getUserType() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    return prefs.getString('userType');
  }
  Future<String?> getUserrType() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      userType = prefs.getString('userType');
    });
    return null;
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        elevation: 0,
        backgroundColor: Colors.white,
        title: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              "N",
              style: GoogleFonts.aBeeZee(
                  textStyle:
                      TextStyle(fontWeight: FontWeight.bold, fontSize: 30)),
            ),
            Image.asset(
              "images/nova.png",
              height: 30,
              width: 30,
            ),
            Text("va",
                style: GoogleFonts.kanit(
                  textStyle: TextStyle(fontSize: 25),
                )),
            Padding(
              padding: const EdgeInsets.only(top: 8.0),
              child: Text(
                "l",
                style: GoogleFonts.sacramento(
                  textStyle: TextStyle(
                      fontSize: 27,
                      fontWeight: FontWeight.bold,
                      color: Colors.amber),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(top: 8.0),
              child: Text(
                "a",
                style: GoogleFonts.sacramento(
                  textStyle: TextStyle(
                      fontSize: 27,
                      fontWeight: FontWeight.bold,
                      color: Colors.redAccent),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(top: 8.0),
              child: Text(
                "bs",
                style: GoogleFonts.sacramento(
                  textStyle: TextStyle(
                      fontSize: 27,
                      fontWeight: FontWeight.bold,
                      color: Colors.blue),
                ),
              ),
            ),
            // Padding(
            //     padding: EdgeInsets.only(right: 5),
            //     child: GestureDetector(
            //       onTap: (){
            //
            //       },
            //       child: CircleAvatar(
            //         radius: 15,
            //         backgroundImage: AssetImage('images/avatar.jpeg'),
            //       ),
            //     ),
            // )
          ],
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 8.0),
            child: GestureDetector(
              onTap: (){
                Navigator.of(context).push(
                  MaterialPageRoute(builder: (context)=>BodyModel())
                );
              },
              child:  CircleAvatar(
                radius: 15,
                backgroundImage: AssetImage('images/avatar.jpeg'),
              ),
            ),
          )
        ],

      ),
      body: Container(
        decoration: BoxDecoration(
          image: DecorationImage(
            image: AssetImage("images/bg.png"),
            fit: BoxFit.cover,
          ),
        ),
        child: SingleChildScrollView(
          // Wrap the entire content in SingleChildScrollView
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.only(top: 20),
                child: Row(
                  children: [
                    // Padding(
                    //   padding: const EdgeInsets.symmetric(
                    //       horizontal: 20, vertical: 10),
                    //   child: Text(
                    //     "AI Dermatologist",
                    //     softWrap: true,
                    //     style: TextStyle(
                    //         color: Colors.white,
                    //         fontWeight: FontWeight.w600,
                    //         fontSize: 30),
                    //   ),
                    // ),
                    // Column(
                    //   children: [
                    //     Text(
                    //       "Your Personal",
                    //       style: TextStyle(
                    //           color: Colors.white,
                    //           fontWeight: FontWeight.w400,
                    //           fontSize: 30),
                    //     ),
                    //     Padding(
                    //       padding: const EdgeInsets.symmetric(horizontal: 20),
                    //       child: Text(
                    //         "AI Dermatologist",
                    //         style: TextStyle(
                    //             color: Colors.white,
                    //             fontWeight: FontWeight.w600,
                    //             fontSize: 30),
                    //       ),
                    //     ),
                    //   ],
                    // )
                  ],
                ),
              ),

              Padding(
                padding: const EdgeInsets.only(top: 20),
                child: Positioned(
                  top: 40,
                  left: 0,
                  right: 0,
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.start,
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      CarouselSlider(
                        options: CarouselOptions(
                          height: 200.0,
                          enlargeCenterPage: true,
                          autoPlay: true,
                          aspectRatio: 16 / 9,
                          autoPlayInterval: Duration(seconds: 3),
                          autoPlayAnimationDuration:
                              Duration(milliseconds: 800),
                          enableInfiniteScroll: true,
                          autoPlayCurve: Curves.fastOutSlowIn,
                          viewportFraction: 0.8,
                        ),
                        items: sliderItems.map((item) {
                          return Builder(
                            builder: (BuildContext context) {
                              return Container(
                                width: MediaQuery.of(context).size.width,
                                margin: EdgeInsets.symmetric(horizontal: 5.0),
                                decoration: BoxDecoration(
                                    image: DecorationImage(
                                        image: AssetImage("images/$item"),
                                        fit: BoxFit.cover),
                                    color: Colors.white,
                                    borderRadius: BorderRadius.circular(10)),
                                child: Center(),
                              );
                            },
                          );
                        }).toList(),
                      ),
                    ],
                  ),
                ),
              ),
              Positioned(
                child: Padding(
                  padding: const EdgeInsets.only(top: 20),
                  child: Container(
                    height: 1200,
                    width: MediaQuery.of(context).size.width * 1,
                    decoration: BoxDecoration(
                      color: Color.fromRGBO(229, 235, 255, 1),
                      borderRadius: BorderRadius.only(
                        topLeft: Radius.circular(30),
                        topRight: Radius.circular(30),
                      ),
                    ),
                    child: Padding(
                      padding: const EdgeInsets.all(15),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            language == 'hindi'
                                ? "पहली पहचान"
                                : "Early Detection",
                            style: GoogleFonts.roboto(
                              textStyle: TextStyle(
                                fontSize: 22,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),

                          Text(
                            language == 'hindi'
                                ? "में अंतर होता है"
                                : "Makes a Difference",
                            style: GoogleFonts.roboto(
                              textStyle: TextStyle(
                                fontSize: 22,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.only(top: 10),
                            child: Container(
                              height: 358,
                              width: MediaQuery.of(context).size.width * 1.0,
                              decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(10),
                              ),
                              child: Column(
                                children: [
                                  Padding(
                                    padding: const EdgeInsets.all(12),
                                    child: Container(
                                      height: 100,
                                      width: MediaQuery.of(context).size.width *
                                          1.0,
                                      decoration: BoxDecoration(
                                        color: Color.fromRGBO(86, 128, 227, 1),
                                        borderRadius: BorderRadius.circular(8),
                                      ),
                                      child: Padding(
                                        padding: const EdgeInsets.all(22),
                                        child: Row(
                                          children: [
                                            Column(
                                              children: [
                                                Text(
                                                  language == 'hindi'
                                                      ? "हमारा टेस्ट आपकी सहायता कर सकता है"
                                                      : "Our test can help you",
                                                  style: GoogleFonts.roboto(
                                                    textStyle: TextStyle(
                                                      fontSize: 19,
                                                      fontWeight:
                                                      FontWeight.w500,
                                                      color: Colors.white,
                                                    ),
                                                  ),
                                                ),
                                                Text(
                                                  language == 'hindi'
                                                      ? "मेलेनोमा का पता लगाने के लिए"
                                                      : "to detect Melanoma.",
                                                  style: GoogleFonts.roboto(
                                                    textStyle: TextStyle(
                                                      fontSize: 19,
                                                      fontWeight:
                                                      FontWeight.w500,
                                                      color: Colors.white,
                                                    ),
                                                  ),
                                                ),
                                              ],
                                            ),
                                            // Padding(
                                            //   padding: const EdgeInsets.only(
                                            //       left: 50),
                                            //   child: Image.asset(
                                            //     "images/scan2.png",
                                            //   ),
                                            // ),
                                          ],
                                        ),
                                      ),
                                    ),
                                  ),
                                  Padding(
                                    padding: const EdgeInsets.all(10),
                                    child: Text(
                                      language == 'hindi'
                                          ? "मेलेनोमा को पहले स्तर पर पहचानना महत्वपूर्ण है; पहली पहचान आपके इलाज के लिए आपकी संभावनाओं को बहुत बढ़ा सकती है।"
                                          : "Finding melanoma at an early stage is crucial; Early detection can vastly increase your chances for cure.",
                                      softWrap: true,
                                      style: GoogleFonts.roboto(
                                        textStyle: TextStyle(
                                          fontSize: 16,
                                          color:
                                          Color.fromRGBO(128, 128, 128, 1),
                                        ),
                                      ),
                                    ),
                                  ),
                                  Padding(
                                    padding: const EdgeInsets.all(10),
                                    child: Text(
                                      language == 'hindi'
                                          ? "बहुत से तिल, भूरे धब्बे और त्वचा पर होने वाले विकास निष्कर्ष होते हैं - लेकिन हमेशा नहीं।"
                                          : "Most moles, brown spots and growths on the skin are harmless - but not always.",
                                      softWrap: true,
                                      style: GoogleFonts.roboto(
                                        textStyle: TextStyle(
                                          fontSize: 16,
                                          color:
                                          Color.fromRGBO(128, 128, 128, 1),
                                        ),
                                      ),
                                    ),
                                  ),
                                  OutlinedButton(
                                    onPressed: () {
                                      // Add your button logic here
                                    },
                                    child: Text(
                                      language == 'hindi'
                                          ? "और पढ़ें"
                                          : "Read More",
                                      style: TextStyle(
                                        fontSize: 16,
                                        color: Colors.black,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.only(top: 10, left: 10),
                            child: Text(
                              language == 'hindi'
                                  ? "आपकी अंतिम स्कैनिंग"
                                  : "Your last scanning",
                              style: GoogleFonts.roboto(
                                textStyle: TextStyle(
                                  fontSize: 22,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ),
                          ),

                          Padding(
                            padding: const EdgeInsets.only(top: 10),
                            child: Container(
                              child: StreamBuilder<QuerySnapshot>(
                                stream: FirebaseFirestore.instance
                                    .collection(userType!)
                                    .doc(FirebaseAuth.instance.currentUser!.email)
                                    .collection("posts")
                                    .snapshots(),
                                builder: (context,
                                    AsyncSnapshot<QuerySnapshot> snapshot) {
                                  if (snapshot.hasData &&
                                      snapshot.data != null) {
                                    List<DocumentSnapshot> nn =
                                        snapshot.data!.docs;
                                    // Calculate the index to get the last two items
                                    int startIndex =
                                        nn.length >= 2 ? nn.length - 2 : 0;

                                    return GridView.builder(
                                      shrinkWrap: true,
                                      physics: NeverScrollableScrollPhysics(),
                                      gridDelegate:
                                          SliverGridDelegateWithFixedCrossAxisCount(
                                        crossAxisCount: 2,
                                        crossAxisSpacing: 10,
                                        mainAxisSpacing: 10,
                                      ),
                                      itemCount: nn.length >= 2
                                          ? 2
                                          : nn.length, // Show the last two items or all if less than two
                                      itemBuilder:
                                          (BuildContext context, int index) {
                                        // Calculate the index based on the last two items
                                        int itemIndex = startIndex + index;

                                        return TextButton(
                                          onPressed: () async {
                                            String? user = await getUserType();
                                            Navigator.of(context).push(
                                                MaterialPageRoute(
                                                    builder: (context) =>
                                                        Details2(
                                                          imagePath: nn[
                                                                  itemIndex]
                                                              ["picture_url"],
                                                          result: nn,
                                                          index: itemIndex,
                                                          usertype: user,
                                                        )));
                                          },
                                          child: Container(
                                            decoration: BoxDecoration(
                                              borderRadius:
                                                  BorderRadius.circular(10),
                                            ),
                                            child: Column(
                                              children: [
                                                Padding(
                                                  padding: EdgeInsets.symmetric(
                                                      vertical: 0),
                                                  child: Container(
                                                    height: 125,
                                                    width: 125,
                                                    decoration: BoxDecoration(
                                                      borderRadius:
                                                          BorderRadius.circular(
                                                              10),
                                                      image: DecorationImage(
                                                        fit: BoxFit.cover,
                                                        image: NetworkImage(
                                                          nn[itemIndex]
                                                              ["picture_url"],
                                                        ),
                                                      ),
                                                    ),
                                                  ),
                                                ),
                                                Padding(
                                                  padding:
                                                      const EdgeInsets.only(
                                                    right: 0,
                                                  ),
                                                  child: Padding(
                                                    padding:
                                                        const EdgeInsets.only(
                                                            top: 10, left: 25),
                                                    child: Column(
                                                      mainAxisAlignment:
                                                          MainAxisAlignment
                                                              .spaceBetween,
                                                      crossAxisAlignment:
                                                          CrossAxisAlignment
                                                              .start,
                                                      children: [
                                                        // Text(
                                                        //   nn[itemIndex]["title"],
                                                        //   textAlign:
                                                        //       TextAlign.start,
                                                        //   style: TextStyle(
                                                        //     fontWeight:
                                                        //         FontWeight.bold,
                                                        //   ),
                                                        // ),
                                                        Row(
                                                          children: [
                                                            Icon(Icons
                                                                .calendar_month),
                                                            Padding(
                                                              padding:
                                                                  const EdgeInsets
                                                                      .only(
                                                                      left: 5),
                                                              child: Text(
                                                                formatTimestamp(
                                                                    nn[itemIndex]
                                                                            [
                                                                            "time"]
                                                                        .seconds),
                                                                style:
                                                                    TextStyle(
                                                                  fontWeight:
                                                                      FontWeight
                                                                          .bold,
                                                                ),
                                                              ),
                                                            ),
                                                          ],
                                                        )
                                                      ],
                                                    ),
                                                  ),
                                                )
                                              ],
                                            ),
                                          ),
                                        );
                                      },
                                    );
                                  } else {
                                    // Handle loading or error state
                                    return CircularProgressIndicator(); // You can replace this with an appropriate loading widget
                                  }
                                },
                              ),
                            ),
                          ),

                          // GridView.builder(
                          //   shrinkWrap: true,
                          //   physics:
                          //       NeverScrollableScrollPhysics(), // Disable GridView's scroll
                          //   gridDelegate:
                          //       SliverGridDelegateWithFixedCrossAxisCount(
                          //     crossAxisCount: 2,

                          //     crossAxisSpacing:
                          //         10, // Add horizontal spacing between grid items
                          //     mainAxisSpacing:
                          //         10, // Number of columns in the grid
                          //   ),
                          //   itemCount: 2, // Number of grid items
                          //   itemBuilder: (BuildContext context, int index) {
                          //     // Create a grid item here
                          //     return Card(
                          //       color: Colors.white,
                          //       margin: EdgeInsets.all(8),
                          //       child: ListTile(
                          //         title: Text('Item $index'),
                          //         onTap: () {
                          //           // Add item tap logic here
                          //         },
                          //       ),
                          //     );
                          //   },
                          // ),

                          //
                          Padding(
                            padding: const EdgeInsets.only(top: 30),
                            child: Container(
                              height: 180,
                              width: MediaQuery.of(context).size.width * 1,
                              decoration: BoxDecoration(
                                  color: Colors.white,
                                  borderRadius: BorderRadius.circular(10)),
                              child: Padding(
                                padding: const EdgeInsets.all(20),
                                child: PieChart(
                                  dataMap: dataMap,
                                  chartType: ChartType.ring,
                                  baseChartColor: Colors.blueAccent,
                                  colorList: colorList,
                                  chartValuesOptions: ChartValuesOptions(
                                    showChartValuesInPercentage: true,
                                  ),
                                  totalValue: 20,
                                ),
                              ),
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.only(top: 10, left: 10),
                            child: Text(
                              language == 'hindi'
                                  ? "आपके लिए रोचक"
                                  : "Interesting for you",
                              style: GoogleFonts.roboto(
                                textStyle: TextStyle(
                                  fontSize: 22,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.all(15),
                            child: Column(
                              children: [
                                CarouselSlider(
                                  items: images.map((image) {
                                    return Builder(
                                      builder: (BuildContext context) {
                                        return Container(
                                          margin: EdgeInsets.symmetric(
                                              horizontal: 5.0),
                                          decoration: BoxDecoration(
                                            image: DecorationImage(
                                              image:
                                                  AssetImage("images/2d.png"),
                                              fit: BoxFit.fitWidth,
                                            ),
                                          ),
                                        );
                                      },
                                    );
                                  }).toList(),
                                  options: CarouselOptions(
                                    height:
                                        180.0, // Adjust the height as needed
                                    // Enable auto-play
                                    enlargeCenterPage: true,
                                    onPageChanged: (index, reason) {
                                      setState(() {
                                        _currentIndex1 = index;
                                      });
                                    },
                                  ),
                                ),
                                Padding(
                                  padding: const EdgeInsets.all(10),
                                  child: DotsIndicator(
                                    dotsCount: images.length,
                                    position: _currentIndex1.toDouble(),
                                    decorator: DotsDecorator(
                                      size: const Size.square(
                                          8.0), // Adjust the dot size
                                      activeSize: const Size(20.0,
                                          8.0), // Adjust the active dot size
                                      activeColor:
                                          Colors.blue, // Color of active dot
                                      spacing: EdgeInsets.all(
                                          4.0), // Adjust the spacing between dots
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          )
                        ],
                      ),
                    ),
                  ),
                ),
              )
            ],
          ),
        ),
      ),

      bottomNavigationBar: (userType == "Doctor")?BottomNavigationBar(
        currentIndex: _currentIndex,
        type: BottomNavigationBarType.fixed,
        selectedItemColor: Colors.blue,
        unselectedItemColor: Colors.grey,
        items: [
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: "",
          ),
          BottomNavigationBarItem(
            icon: IconButton(
              icon: Icon(Icons.notifications),
              onPressed: () {
                print(FirebaseAuth.instance.currentUser!.email);
                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (context) => NotificationScreen(),
                  ),
                );
              },
            ),
            label: "",
          ),
          BottomNavigationBarItem(
            icon: SizedBox(
              width: 60.0,
              height: 60.0,
              child: FloatingActionButton(
                onPressed: () async {
                  String? userType = await getUserType();
                  _showMyDialog(cameras, userType, context);
                },
                child: Icon(
                  Icons.camera,
                  size: 36.0,
                ),
                backgroundColor: Colors.blue,
              ),
            ),
            label: "",
          ),
          BottomNavigationBarItem(
            icon: IconButton(
              icon: Icon(Icons.book),
              onPressed: () async {
                String? user = await getUserType();
                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (context) => Report(user: user),
                  ),
                );
              },
            ),
            label: "",
          ),
          BottomNavigationBarItem(
            icon: IconButton(
              icon: Icon(Icons.person),
              onPressed: () {
                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (context)=>IotDevice(),
                  ),
                );
              },
            ),
            label: "",
          ),
        ],
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
      ):
      BottomNavigationBar(
        currentIndex: _currentIndex,
        type: BottomNavigationBarType.fixed,
        selectedItemColor: Colors.blue,
        unselectedItemColor: Colors.grey,
        items: [
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: "",
          ),
          BottomNavigationBarItem(
            icon: IconButton(
              icon: Icon(Icons.notifications),
              onPressed: () {
                print(FirebaseAuth.instance.currentUser!.email);
                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (context) => Dummy(),
                  ),
                );
              },
            ),
            label: "",
          ),
          BottomNavigationBarItem(
            icon: SizedBox(
              width: 60.0,
              height: 60.0,
              child: FloatingActionButton(
                onPressed: () async {
                  String? userType = await getUserType();
                  _showMyDialog(cameras, userType, context);
                },
                child: Icon(
                  Icons.camera,
                  size: 36.0,
                ),
                backgroundColor: Colors.blue,
              ),
            ),
            label: "",
          ),
          BottomNavigationBarItem(
            icon: IconButton(
              icon: Icon(Icons.book),
              onPressed: () async {
                String? user = await getUserType();
                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (context) => Report(user: user),
                  ),
                );
              },
            ),
            label: "",
          ),
          BottomNavigationBarItem(
            icon: IconButton(
              icon: Icon(Icons.person),
              onPressed: () {
                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (context)=>IotDevice(),
                  ),
                );
              },
            ),
            label: "",
          ),
        ],
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
      )
    );
  }

  Future<void> _showMyDialog(List<CameraDescription>? cameras, String? type,
      BuildContext context) async {
  
    return showDialog<void>(
      context: context,
      barrierDismissible: false, // user must tap button!
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text(
            language=='hindi'?'आपक्षित क्षेत्र':
            'Guidelines to be followed before taking picture',
            style: GoogleFonts.poppins(fontWeight: FontWeight.bold),
          ),
          content: SingleChildScrollView(
            child: Container(
              child: ListBody(
                children: <Widget>[
                    Text("1. Use good, natural light for accurate colors."),
                    Padding(
                      padding: const EdgeInsets.only(top: 3),
                      child: Text("2. Keep the camera steady for clear shots."),
                    ),
                    Padding(
                      padding: const EdgeInsets.only(top: 3),
                      child: Text("3. Capture close-ups for detail."),
                    ),
                    Padding(
                      padding: const EdgeInsets.only(top: 3),
                      child: Text("4. Use a plain, consistent background."),
                    ),
                    Padding(
                      padding: const EdgeInsets.only(top: 3),
                      child: Text("5. Avoid using filters or edits."),
                    ),
                    Padding(
                      padding: const EdgeInsets.only(top: 3),
                      child: Text("6. Maintain even lighting conditions."),
                    ),
                    Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Container(
                        width:200,
                        child:OutlinedButton(
                          onPressed: (){

                            Navigator.pop(context,'Ok');
                            _showMyDialog1(cameras, type, context);
                          },
                          child:Text("Next")
                        )
                      ),
                    )
                ],
              ),
            ),
          ),
        );
      },
    );
  }
  Future<void> _showMyDialog1(List<CameraDescription>? cameras, String? type,
      BuildContext context) async {
    bool _isloading = false;
    String dropdown = listval1.first;
    return showDialog<void>(
      context: context,
      barrierDismissible: false, // user must tap button!
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text(
            language=='hindi'?'आपक्षित क्षेत्र':
            'Your Affected region',
            style: GoogleFonts.poppins(fontWeight: FontWeight.bold),
          ),
          content: SingleChildScrollView(
            child: Container(
              child: ListBody(
                children: <Widget>[
                  DropdownMenu<String>(
                    initialSelection: listval1.first,
                    width: MediaQuery.of(context).size.width * 0.67,
                    onSelected: (String? value) {
                      // This is called when the user selects an item.
                      setState(() {
                        dropdown = value!;
                      });
                    },
                    dropdownMenuEntries:
                        listval1.map<DropdownMenuEntry<String>>((String value) {
                      return DropdownMenuEntry<String>(
                          value: value, label: value);
                    }).toList(),
                  ),
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Container(
                      decoration: BoxDecoration(
                          color: Color.fromRGBO(86, 128, 227, 1),
                          borderRadius: BorderRadius.circular(20)),
                      child: TextButton(
                        onPressed: () async {

                          Navigator.of(context).push(MaterialPageRoute(
                              builder: (context) => Questions( cameras: cameras, user: type, skin: dropdown)));

                          // _showMyDialog2(cameras, type, context, dropdown, (bool isLoading) {
                          //   setState(() {
                          //     _isloading = isLoading;
                          //   });
                          // });

                        },
                        child: Text(
                          language=='hindi'?"आगे":
                          "Next",
                          style: GoogleFonts.poppins(
                              color: Colors.white,
                              fontSize: 16,
                              fontWeight: FontWeight.bold),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

}

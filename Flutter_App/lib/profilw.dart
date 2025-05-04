import 'package:camera/camera.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:novalabs/GoogleSignin.dart';

import 'package:persistent_bottom_nav_bar/persistent_tab_view.dart';


class Profile extends StatefulWidget {
  const Profile({super.key});

  @override
  State<Profile> createState() => _ProfileState();
}

class _ProfileState extends State<Profile> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        actions: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Icon(
              Icons.person_pin_circle_rounded,
              size: 33,
              color: Color.fromARGB(255, 241, 194, 125),
            ),
          )
        ],
        elevation: 0,
        backgroundColor: Colors.white,
        title: Row(
          crossAxisAlignment: CrossAxisAlignment.center,
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
          ],
        ),
      ),
      body: Center(
        child: TextButton(
          onPressed: () async {
            await FirebaseAuth.instance.signOut();
            PersistentNavBarNavigator.pushNewScreen(
              context,
              screen: GoogleSignin(),
              withNavBar: false,
              pageTransitionAnimation: PageTransitionAnimation.fade,
            );
          },
          child: Text("Logout"),
        ),
      ),
    );
  }
}

-- phpMyAdmin SQL Dump
-- version 5.1.3
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: May 18, 2022 at 04:07 PM
-- Server version: 10.4.22-MariaDB
-- PHP Version: 7.4.28

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `emotion2`
--

-- --------------------------------------------------------

--
-- Table structure for table `analyse`
--

CREATE TABLE `analyse` (
  `id` int(11) NOT NULL,
  `stid` int(11) NOT NULL,
  `emo` varchar(20) NOT NULL,
  `emoval` int(11) NOT NULL,
  `dt` date NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `analyse`
--

INSERT INTO `analyse` (`id`, `stid`, `emo`, `emoval`, `dt`) VALUES
(1, 4, 'Neutral', 1, '2021-12-12'),
(2, 4, 'Happy', 1, '2021-12-12'),
(3, 4, 'Tensed', 0, '2021-12-12'),
(4, 4, 'Happy', 1, '2021-12-12'),
(5, 4, 'Neutral', 1, '2021-12-12'),
(6, 4, 'Happy', 1, '2021-12-12'),
(7, 4, 'Neutral', 1, '2021-12-12'),
(8, 4, 'Happy', 1, '2021-12-12'),
(9, 4, 'Fearful', 0, '2021-12-12'),
(10, 4, 'Happy', 1, '2021-12-12'),
(11, 4, 'Neutral', 1, '2021-12-12'),
(12, 4, 'Happy', 1, '2021-12-12'),
(13, 1, 'Neutral', 230, '2022-05-12'),
(14, 1, 'Tensed', 226, '2022-05-12'),
(15, 1, 'Neutral', 227, '2022-05-12'),
(16, 1, 'Happy', 231, '2022-05-12'),
(17, 1, 'Tensed', 197, '2022-05-12'),
(18, 1, 'Happy', 260, '2022-05-12'),
(19, 1, 'Stressed', 234, '2022-05-12'),
(20, 1, 'Happy', 279, '2022-05-12'),
(21, 1, 'Stressed', 266, '2022-05-12'),
(22, 1, 'Tensed', 262, '2022-05-12'),
(23, 1, 'Neutral', 268, '2022-05-12');

-- --------------------------------------------------------

--
-- Table structure for table `picdata`
--

CREATE TABLE `picdata` (
  `id` int(11) NOT NULL,
  `finfo` varchar(200) NOT NULL,
  `pic` varchar(200) NOT NULL,
  `dt` date NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `picdata`
--

INSERT INTO `picdata` (`id`, `finfo`, `pic`, `dt`) VALUES
(1, 'case no1', '01.jpg', '2022-05-12');

-- --------------------------------------------------------

--
-- Table structure for table `staff_data`
--

CREATE TABLE `staff_data` (
  `id` int(11) NOT NULL,
  `nme` varchar(150) NOT NULL,
  `addr` varchar(250) NOT NULL,
  `con` varchar(15) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `staff_data`
--

INSERT INTO `staff_data` (`id`, `nme`, `addr`, `con`) VALUES
(1, 'Arun', 'Arunodayam\r\nKilimanoor', '9446569874'),
(2, 'Rajesh', 'R R Bhavan\r\nKilimanoor', '9446569878'),
(3, 'Gopal', 'Gopal bhavan\r\nKilimanoor', '9446563005'),
(4, 'Vineeth', 'v v bhavan\r\nKilimanoor', '9446562005'),
(5, 'Arun', 'Kilimanoor', '9995878787');

-- --------------------------------------------------------

--
-- Table structure for table `tips_data`
--

CREATE TABLE `tips_data` (
  `id` int(11) NOT NULL,
  `tips` varchar(250) NOT NULL,
  `st` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `tips_data`
--

INSERT INTO `tips_data` (`id`, `tips`, `st`) VALUES
(1, 'Relax and do work perfectly', 1);

-- --------------------------------------------------------

--
-- Table structure for table `user_log`
--

CREATE TABLE `user_log` (
  `id` int(11) NOT NULL,
  `uid` varchar(50) NOT NULL,
  `pas` varchar(50) NOT NULL,
  `typ` varchar(10) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `user_log`
--

INSERT INTO `user_log` (`id`, `uid`, `pas`, `typ`) VALUES
(1, 'admin', 'admin', 'admin'),
(2, 'k101', 'asd', 'ao'),
(3, 'appu123', 'test', 'far'),
(4, '9446562005', 'stf', 'stf'),
(5, '9995878787', 'stf', 'stf');

-- --------------------------------------------------------

--
-- Table structure for table `vdodata`
--

CREATE TABLE `vdodata` (
  `id` int(11) NOT NULL,
  `finfo` varchar(200) NOT NULL,
  `pic` varchar(200) NOT NULL,
  `dt` date NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `vdodata`
--

INSERT INTO `vdodata` (`id`, `finfo`, `pic`, `dt`) VALUES
(1, 'check video', '4.mp4', '2022-05-18');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `analyse`
--
ALTER TABLE `analyse`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `picdata`
--
ALTER TABLE `picdata`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `staff_data`
--
ALTER TABLE `staff_data`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `tips_data`
--
ALTER TABLE `tips_data`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `user_log`
--
ALTER TABLE `user_log`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `vdodata`
--
ALTER TABLE `vdodata`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `analyse`
--
ALTER TABLE `analyse`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=24;

--
-- AUTO_INCREMENT for table `picdata`
--
ALTER TABLE `picdata`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `staff_data`
--
ALTER TABLE `staff_data`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- AUTO_INCREMENT for table `tips_data`
--
ALTER TABLE `tips_data`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `user_log`
--
ALTER TABLE `user_log`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- AUTO_INCREMENT for table `vdodata`
--
ALTER TABLE `vdodata`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
